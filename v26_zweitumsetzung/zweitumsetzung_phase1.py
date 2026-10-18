#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zweitumsetzung der elf neuen Fassungen der Indikatorrevision V2.6 auf Phase 1 (Protokoll, Abschnitt 3.2).

Aufruf:
    python3 zweitumsetzung_phase1.py liste     nennt die Teile in der Reihenfolge, in der sie laufen muessen
    python3 zweitumsetzung_phase1.py <teil>    rechnet einen Teil und speichert sein Zwischenergebnis

Teile: basis, phrasen, q2, eo3, eo4, eo5, ds2, di1, di4_ip2, zusammen.
eo5 und di1 rechnen in Bloecken von Topics. Reicht das Zeitbudget eines Aufrufs nicht, meldet der Teil
status=fortsetzen; dann denselben Teil erneut aufrufen, er setzt beim naechsten fehlenden Block fort.

Einzige fachliche Quellen: V2.6_Protokoll.md (Abschnitte 0.4, 1b bis 1g, Aenderungen an diesem Protokoll),
schnittstelle.md und hyland_lexikon.json. Massgeblich ist der Stand nach E9.

Pfade: Das Skript liegt in V26/v26_zweitumsetzung/; alle Eingaben relativ zu V26; alle Zwischen- und
Endergebnisse in V26/output_v26_zweitumsetzung/ (Zwischenergebnisse im Unterordner zwischen/, nur Zahlen).
Konsole und Ausgabedateien enthalten nur Zahlen und Bezeichner, keine Texte aus den Daten.

Lesarten: Die Liste LESARTEN direkt unter diesem Kopf haelt jede Stelle fest, an der das Protokoll mehrdeutig
oder unvollstaendig ist, mit der gewaehlten (woertlichsten) Lesart. Sie wird auch in die Info-Datei geschrieben.
Eintraege mit dem Zusatz "nach Praezisierung vom 17.10." folgen dem letzten Aenderungseintrag des Protokolls
("Zweitumsetzung nach 3.2, erster Abgleich, und Praezisierungen", Punkte 1 bis 9; Protokoll 6d16fa5f).

Laeuft mit Python 3.10, numpy, pandas, scipy, scikit-learn und der Standardbibliothek; kein Netz.
"""

# ---------------------------------------------------------------------------------------------------------------
# Lesarten
# ---------------------------------------------------------------------------------------------------------------
LESARTEN = [
    # Allgemein
    "ALLG-1 Korpus: alle Zeilen von df nach der Ladevorschrift (20.706), Rauschdokumente (Label -1) eingeschlossen."
    " Wo das Protokoll 'Korpus' oder 'alle Dokumente' sagt (Vokabular, Erstjahr der Phrasen, Jahrgangsmittel bei"
    " DS2 A3, Jahresmittel und Perzentile der Zitationen, Dezilpaare bei EO3 A1, Dokumente ausserhalb des Topics"
    " bei EO5 A1), sind diese Zeilen gemeint. Topicdokumente sind die Zeilen mit dem Label des Topics; je Topic"
    " alle seine Dokumente, keine Pseudokorpora.",
    "ALLG-2 Ladevorschrift woertlich auf die Kopfzeile der Schnittstelle (23 Spalten); gelesen werden nur die"
    " benoetigten Spalten Title, Year, Abstract, Author Full Names, Times Cited, WoS Core, Keywords Plus,"
    " WoS Categories, Affiliations und ORCIDs (usecols), geprueft gegen die Kopfzeile vor dem Lesen. Das aendert"
    " weder Zeilen noch Reihenfolge. Die Umbenennung der Ladevorschrift wirkt nur auf vorhandene Spalten; danach"
    " heisst Author Full Names 'Authors' (volle Namen) und Times Cited, WoS Core 'Times Cited'.",
    "ALLG-3 (nach Praezisierung vom 17.10., Punkte 7 und 8) Feldeintraege (WoS Categories, Keywords Plus,"
    " Affiliations, Authors, ORCIDs): Zelle an ';' getrennt, Randleerraum entfernt, leere Eintraege verworfen (bei"
    " der Exporttrennung '; ' dasselbe wie eine Trennung an '; '). EO2 A1: Kategorien in Originalschreibung,"
    " Mehrfachnennungen als Zaehlwert x_dk. DI2 A1: Institutionen je Dokument als Menge (Originalschreibung)."
    " IP3 A1: Keywords Plus kleingeschrieben, je Dokument als Menge. Autoren: siehe DI1-1 und DI1-2.",
    "ALLG-4 Kosinusaehnlichkeit: Einbettungen in float64 umgewandelt und je Zeile auf Laenge 1 normiert.",
    "ALLG-5 (nach Praezisierung vom 17.10., Punkt 9, fuer DI4) NaN nur, wo das Protokoll eine Mindestzahl nennt"
    " (EO4 A2b 20 Woerter, EO5 A1 20 Dokumente und 5 Woerter, DI4 A4 3 Dokumente und positive Summe von x, IP2 A4"
    " 4 Dokumente aus 2 Jahren) oder wo der Ausdruck mathematisch unbestimmt ist (Nenner 0, kein Dokument mit Wert,"
    " lambda = 0). Sonst keine zusaetzlichen Mindestzahlen.",
    "ALLG-6 'rarefiziert auf m Dokumente' bei den q = 2-Massen (EO1 A1, EO2 A1, DI2 A1, IP3 A1): keine Ziehung,"
    " lambda auf allen Dokumenten des Topics, weil die Rarefaktion laut 1b analytisch erledigt ist. Bei DI1 A1"
    " dagegen echte Ziehungen (1b).",
    "ALLG-7 Wo 0.4 und 1b verschieden benennen, gilt die Rechenvorschrift aus 1b: EO1 A1 und IP3 A1 als -ln lambda"
    " (nicht Hill-Zahl 1/lambda), EO2 A1 als -ln lambda / ln 99 (nicht inverse Simpson), DI2 A1 als lambda.",
    # Phrasen
    "PHR-1 Phrasentext ist die Spalte text der Ladevorschrift (Titel + '. ' + Abstract), kleingeschrieben; der"
    " eingefuegte Punkt trennt Titel und Abstract. Ohne Bereinigung der Verlagsvermerke (Stand nach E9).",
    "PHR-2 (nach Praezisierung vom 17.10., Punkt 1) Abschnitte: Der kleingeschriebene Text wird an jeder Folge der"
    " Zeichen . , ; : ! ? ( ) [ ] { } \" / geteilt (re.split mit [.,;:!?()\\[\\]{}\"/]+); innerhalb eines Abschnitts"
    " trennt Leerraum (str.split) die Rohwoerter. Von jedem Rohwort fallen vorn und hinten Bindestriche und"
    " Apostrophe weg (strip mit ASCII '-' und ASCII-Apostroph; typografische Apostrophe und Gedankenstriche bleiben"
    " stehen und machen das Rohwort zum Nicht-Inhaltswort).",
    "PHR-3 (nach Praezisierung vom 17.10., Punkt 1) Inhaltswort: ein Rohwort, das nach dem Abschneiden mindestens"
    " zwei Zeichen hat, kein Stoppwort ist und ganz dem Muster [a-z][a-z0-9]*(?:-[a-z0-9]+)* entspricht (fullmatch;"
    " also kein Zahlwort, kein Wort mit Ziffer am Anfang, kein Wort mit Nicht-ASCII-Zeichen). Jedes andere Rohwort"
    " (auch ein nach dem Abschneiden leeres) und das Abschnittsende beenden den Lauf.",
    "PHR-4 (nach Praezisierung vom 17.10., Punkt 1) Stoppwoerter: sklearn ENGLISH_STOP_WORDS (318 Woerter), Vergleich"
    " mit dem abgeschnittenen Rohwort.",
    "PHR-5 Unigramme: alle Inhaltswoerter; Bigramme: je zwei im Lauf benachbarte Inhaltswoerter, mit einem"
    " Leerzeichen verbunden. Je Dokument eine Menge aus Uni- und Bigrammen.",
    "PHR-6 Vokabular: Dokumentfrequenz ueber alle Korpusdokumente; aufgenommen bei mindestens 2 und hoechstens"
    " 0,25 mal Korpusgroesse Dokumenten (bei 20.706 also hoechstens 5.176). Unigramme und Bigramme gleich behandelt.",
    "PHR-7 EO1 A1 nutzt nur die Vokabularphrasen (das Protokoll nennt das Vokabular als Grundlage der Phrasen).",
    # EO1 A1
    "EO1-1 lambda = (Sum n_k^2 - Sum x_dk^2) / (N^2 - Sum L_d^2) ueber die Vokabularphrasen aller Topicdokumente,"
    " x_dk binaer (also Sum x_dk^2 = N); EO1_A1 = -ln lambda; NaN bei Nenner 0 oder lambda = 0 (-ln 0 unendlich)."
    " Bei DI2_A1 = lambda ist lambda = 0 dagegen ein Wert.",
    # EO2 A1
    "EO2-1 Eintraege aus WoS Categories (ALLG-3); EO2_A1 = -ln lambda / ln 99 mit der Konstante 99 aus 1b, auch"
    " wenn die beobachtete Kategorienzahl abweicht (sie steht in der Info-Datei). Nicht gewaehlt: Normierung an der"
    " beobachteten Kategorienzahl.",
    # EO3 A1
    "EO3-1 Beobachtung: exakte mittlere Kosinusaehnlichkeit ueber alle ungeordneten Paare verschiedener"
    " Topicdokumente (ueber die Summenidentitaet, gleich der Paarschleife).",
    "EO3-2 Textlaenge: Zahl der Woerter von text_clean (text_clean.split(), wie im Laengenfilter der"
    " Ladevorschrift), also Titel und Abstract, gezaehlt in Woertern, nicht in Zeichen.",
    "EO3-3 (nach Praezisierung vom 17.10., Punkt 2) Klassen: Grenzen sind die Quantile 0,1 bis 0,9 der Textlaenge"
    " ueber alle Korpusdokumente (numpy.quantile, lineare Interpolation), gleiche Grenzen zusammengefasst"
    " (numpy.unique); eine Laenge auf einer Grenze gehoert zur oberen Klasse (searchsorted mit side='right').",
    "EO3-4 Erwartung: fuer jedes Dezilpaar (a, b) die mittlere Kosinusaehnlichkeit aller Korpuspaare mit einem"
    " Dokument in a und einem in b (fuer a = b alle Paare verschiedener Dokumente in a), gemittelt ueber alle Paare"
    " des Topics nach deren Dezilpaar. Das ist der Erwartungswert fuer eine Zufallsmenge gleicher Groesse und"
    " gleicher Laengenzusammensetzung. Alle Korpuspaare, auch die des Topics selbst. Nicht gewaehlt: ein Wert je"
    " Dezil nur aus den Paaren innerhalb des Dezils, gemittelt nach den Dezilen der Topicdokumente.",
    "EO3-5 EO3_A1 = Erwartung - Beobachtung der Kosinusaehnlichkeit (1b; hoeher heisst breiter gestreut). Das"
    " 'beobachtet minus erwartet' aus 0.4 bezieht sich auf die Unaehnlichkeit 1 - Kosinus wie in v2.5 und ergibt"
    " denselben Wert.",
    # EO4 A1b
    "EO4-1 Bewertete Stuecke: Stuecke aus saetze_phase1.jsonl mit len(s.split()) >= 4, in Dokument- und"
    " Stueckreihenfolge; das k-te gehoert zu werte[k]. Geprueft gegen saetze_meta.npz (doc gleich, nr bis auf eine"
    " feste Verschiebung gleich, 0- oder 1-basiert); bei Abweichung bricht der Teil ab.",
    "EO4-2 (nach Praezisierung vom 17.10., Punkt 4) Vermerk, jeweils ohne Ruecksicht auf die Schreibung: (a) das"
    " Stueck beginnt nach fuehrendem Leerraum mit '(c)', dem Zeichen Copyright oder dem Wort 'copyright'"
    " (copyright\\b); oder (b) es hat hoechstens 20 Woerter (split, wie bisher) und enthaelt mit Wortgrenzen"
    " 'all rights reserved', 'published by' mit irgendwo danach einem der Woerter elsevier, wiley, springer, press,"
    " society, institute oder publishing, 'elsevier' gefolgt (nach Leerraum) von b.v., bv, b v, b.v (Muster"
    " b\\.?\\s?v), ltd, inc, science, masson oder gmbh, 'open access article' oder 'creative commons'. Die"
    " codeartig gesetzten Wortfolgen gelten mit einfachen Leerzeichen. Die Pruefung gilt fuer alle Stuecke, auch"
    " die unter 4 Woertern (wie bisher).",
    "EO4-3 (nach Praezisierung vom 17.10., Punkt 3) Befundsatz: das Stueck enthaelt ohne Ruecksicht auf die"
    " Schreibung als ganzes Wort (\\b) eine der Formen show, shows, showed, shown, find, finds, found, demonstrate,"
    " demonstrates, demonstrated, observe, observes, observed, reveal, reveals, revealed, indicate, indicates,"
    " indicated, suggest, suggests, suggested, conclude, concludes, concluded, confirm, confirms, confirmed, prove,"
    " proves, proved, proven, result oder results; Formen auf -ing und resulted zaehlen nicht.",
    "EO4-4 EO4 A1b je Dokument: minus das Mittel der Modellwerte seiner bewerteten Befundstuecke ohne Vermerk; ohne"
    " solches minus der Wert seines letzten bewerteten Stuecks ohne Vermerk; ohne bewertetes Stueck ohne Vermerk"
    " kein Wert. Topicwert = Mittel ueber die Topicdokumente mit Wert. Modellwerte (float32) in float64 gemittelt.",
    # EO4 A2b
    "EO4-5 EO4 A2b: Text ist der Abstract allein (ohne Titel). Hat der Abstract mindestens ein Vermerkstueck, dann"
    " die uebrigen Stuecke mit Leerzeichen verbunden, sonst der Abstract unveraendert (so gleich 1b bei Abstracts"
    " ohne Vermerk).",
    "EO4-6 (nach Praezisierung vom 17.10., Punkt 5) Woerter sind die Treffer von [a-z]+(?:'[a-z]+)? im"
    " kleingeschriebenen Text (ohne Umwandlung typografischer Apostrophe); ihre Zahl ist die Wortzahl. Dokumentwert"
    " nur ab 20 solchen Woertern.",
    "EO4-7 (nach Praezisierung vom 17.10., Punkt 5) Zaehlung von links nach rechts ueber diese Woerter: an jeder"
    " Stelle zuerst der laengste passende Mehrwortausdruck aus Hedge- und Boosterliste zusammen; er zaehlt fuer"
    " seine Liste, und seine Woerter zaehlen nicht noch einmal (es geht hinter ihm weiter); sonst zaehlt das Wort,"
    " wenn es in der Einwortliste der Hedges steht. Booster (Mehrwort wie oben, Einwort an nicht verbrauchten"
    " Stellen) nur zur Information, nicht im Wert.",
    "EO4-8 Dokumentdichte = 1000 * Hedges / Woerter; Topicwert = Mittel der Dokumentdichten (nicht der gepoolte"
    " Quotient), wie bei A1 'Mittel ueber die Dokumente des Topics'.",
    # EO4 R3b
    "EO4-9 EO4 R3b = (z(A1b) + z(A2b)) / 2, z je Fassung ueber die Topics mit Wert, mit Mittel und"
    " Standardabweichung mit ddof = 0 (die Topics sind die Gesamtheit); NaN, wenn eine der beiden fehlt. Nicht"
    " gewaehlt: ddof = 1 (R3b waere dann bei 146 Topics um den festen Faktor sqrt(145/146) = 0,99657 kleiner, Raenge"
    " gleich).",
    # EO5 A1
    "EO5-1 Woerter sind die Unigramme des Phrasenvokabulars (PHR-1 bis PHR-6, ohne Bereinigung), je Dokument als"
    " Menge, jedes Wort einmal je Dokument.",
    "EO5-2 Dokumente mit w ausserhalb von t: alle Korpusdokumente mit w, deren Label nicht t ist (Rauschen und"
    " andere Topics eingeschlossen).",
    "EO5-3 c-quer = Mittel der Kosinusaehnlichkeit ueber alle ungeordneten Paare verschiedener solcher Dokumente;"
    " gueltig ab 20 solchen Dokumenten; SemD = -ln(max(c-quer, 0,001)).",
    "EO5-4 Dokumentwert = Mittel von SemD ueber die gueltigen Woerter des Dokuments, nur ab 5 gueltigen Woertern;"
    " Topicwert = Mittel ueber diese Dokumente.",
    # DS2 A3
    "DS2-1 Erstjahr einer Vokabularphrase = kleinstes Erscheinungsjahr unter allen Korpusdokumenten mit der Phrase.",
    "DS2-2 Anteil je Dokument mit Jahr ab 2003 und mindestens einer Vokabularphrase: Anteil seiner Vokabularphrasen"
    " mit Erstjahr >= Jahr - 2. Jahrgangsmittel = Mittel dieses Anteils ueber alle Korpusdokumente desselben Jahres"
    " mit Anteil. Dokumentwert = Anteil - Jahrgangsmittel; Topicwert = Mittel ueber die Topicdokumente mit Wert.",
    # DI1 A1
    "DI1-1 (nach Praezisierung vom 17.10., Punkt 6) Namen aus Author Full Names (nach der Umbenennung Authors): an"
    " ';' getrennt, Randleerraum entfernt, kleingeschrieben, Leerraum auf einfache Leerzeichen gebracht"
    " (' '.join(name.lower().split())); leere Namen verworfen.",
    "DI1-2 (nach Praezisierung vom 17.10., Punkt 6) ORCIDs: Eintraege 'Name/ID' am letzten '/' getrennt, der Name"
    " ebenso normiert, die ID ohne Randleerraum; bei gleichem normierten Namen gilt der letzte Eintrag. Kennung ist"
    " die ORCID, wenn der normierte Name im selben Dokument zugeordnet ist, sonst der normierte Name. Eine ORCID wird"
    " nicht auf andere Dokumente mit demselben Namen uebertragen (wie bisher).",
    "DI1-3 (nach Praezisierung vom 17.10., Punkt 6) Ziehung unter den Topicdokumenten mit mindestens einem Autor,"
    " ohne Zuruecklegen, gleichverteilt; m = min(25, kleinste Zahl solcher Dokumente ueber die Topics). Zwei"
    " Dokumente sind verbunden, wenn sie eine Kennung teilen; gezaehlt werden die Komponenten der m Dokumente."
    " Wert = 1 - (Komponenten - 1) / (m - 1).",
    "DI1-4 1.000 Ziehungen je Topic mit numpy default_rng(Seed-Folge [20261017, Topic]); DI1_A1 = Mittel,"
    " DI1_A1_SD = Standardabweichung der Einzelziehungen mit ddof = 1.",
    # DI2 A1
    "DI2-1 (nach Praezisierung vom 17.10., Punkt 7) Institutionen = Eintraege der Spalte Affiliations (ALLG-3), je"
    " Dokument als Menge, nicht aus Addresses abgeleitet; DI2_A1 = lambda.",
    # DI4 A4
    "DI4-1 (nach Praezisierung vom 17.10., Punkt 9) Zitationen = Spalte Times Cited (Times Cited, WoS Core),"
    " numerisch; fehlende Zitationszahlen zaehlen als 0, auch im Jahrgangsmittel ueber alle Korpusdokumente.",
    "DI4-2 x = Zitationen / Mittel der Zitationen aller Korpusdokumente desselben Erscheinungsjahrs (x = 0, falls"
    " dieses Mittel 0 ist).",
    "DI4-3 (nach Praezisierung vom 17.10., Punkt 9) sigma = Standardabweichung von ln(1 + x) ueber alle"
    " Topicdokumente mit n - 1 im Nenner (ddof = 1); DI4_A4 = 2 Phi(sigma / sqrt 2) - 1; NaN unter 3 Dokumenten"
    " oder wenn die Summe von x im Topic nicht positiv ist.",
    # IP2 A4
    "IP2-1 Zitationsperzentil je Jahrgang = mittlerer Rang (Bindungen gemittelt) der Zitationen unter allen"
    " Korpusdokumenten desselben Erscheinungsjahrs, geteilt durch deren Zahl (wie pandas rank(pct=True)). Nicht"
    " gewaehlt: (Rang - 0,5)/n wie scipy percentileofscore(kind='mean'); tau_b vergleicht Perzentile ueber Jahre"
    " hinweg und haengt daher von dieser Wahl ab. Dokumente ohne Zitationszahl fallen bei IP2 A4 heraus"
    " (unveraendert; Punkt 9 der Praezisierung betrifft nur DI4 A4).",
    "IP2-2 IP2_A4 = Kendall tau_b (scipy.stats.kendalltau) zwischen Erscheinungsjahr und Perzentil ueber die"
    " Topicdokumente; NaN unter 4 Dokumenten, unter 2 verschiedenen Jahren oder wenn tau_b unbestimmt ist.",
    # IP3 A1
    "IP3-1 (nach Praezisierung vom 17.10., Punkt 8) Eintraege aus Keywords Plus (ALLG-3), kleingeschrieben"
    " (str.lower), je Dokument als Menge; IP3_A1 = -ln lambda.",
]

import gc
import hashlib
import json
import math
import os
import pickle
import re
import resource
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

T_START = time.perf_counter()

# ---------------------------------------------------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------------------------------------------------
SKRIPT = Path(__file__).resolve()
V26 = SKRIPT.parent.parent
AUS = V26 / "output_v26_zweitumsetzung"
ZW = AUS / "zwischen"
P_CSV = V26 / "output_neulauf_v2.5" / "wos_qc_phase1_2000_2015.csv"
P_PKL = V26 / "output_neulauf_v2.5" / "prior" / "output_phase1" / "model_results.pkl"
P_WERTE = V26 / "output_v26_zertifiziertheit" / "phase1" / "werte.npy"
P_META = V26 / "output_v26_zertifiziertheit" / "phase1" / "saetze_meta.npz"
P_SAETZE = AUS / "eingaben" / "saetze_phase1.jsonl"
P_LEX = SKRIPT.parent / "hyland_lexikon.json"
P_CSV_AUS = AUS / "zweitumsetzung_phase1.csv"
P_INFO_AUS = AUS / "zweitumsetzung_phase1_info.json"

# ---------------------------------------------------------------------------------------------------------------
# Parameter (Protokoll und Schnittstelle)
# ---------------------------------------------------------------------------------------------------------------
M_RAREF = 25                 # 0.4: m = 25
DI1_ZIEHUNGEN = 1000         # Schnittstelle: 1.000 Ziehungen je Topic
DI1_SEED = 20261017          # eigener fester Seed
N_KAT_NORM = 99              # 1b: ln 99
VOK_MIN_DF = 2               # 1b: in mindestens 2 Dokumenten
VOK_MAX_ANTEIL = 0.25        # 1b: in hoechstens 25 % der Dokumente
EO5_MIN_DOK = 20             # Aenderungseintrag EO5: gueltig ab 20 Dokumenten
EO5_MIN_WOERTER = 5          # mindestens 5 gueltige Woerter
EO5_C_MIN = 0.001            # max(c, 0,001)
EO4_MIN_WOERTER = 20         # 1b: ab 20 Woertern
SATZ_MIN_WOERTER = 4         # Schnittstelle: bewertet ab 4 Woertern
VERMERK_MAX_WOERTER = 20     # Bereinigung: Stueck mit hoechstens 20 Woertern
DS2_AB_JAHR = 2003           # DS2 A3: je Dokument ab 2003
DS2_FENSTER = 2              # Erstjahr mindestens Erscheinungsjahr minus 2
DI4_MIN_DOK = 3              # DI4: mit mindestens 3 Dokumenten
IP2_MIN_DOK = 4              # IP2 A4: mindestens 4 Dokumente
IP2_MIN_JAHRE = 2            # aus mindestens 2 Jahren
ZEITBUDGET_S = 100.0         # danach setzen blockweise Teile beim naechsten Aufruf fort
BLOCK_TOPICS = 20            # Topics je Block (eo5, di1)

SPALTEN_AUSGABE = ["EO1_A1", "EO2_A1", "EO3_A1", "EO4_A1B", "EO4_A2B", "EO4_R3B", "EO5_A1", "DS2_A3", "DI1_A1",
                   "DI1_A1_SD", "DI2_A1", "DI4_A4", "IP2_A4", "IP3_A1"]

TEILE = ["basis", "phrasen", "q2", "eo3", "eo4", "eo5", "ds2", "di1", "di4_ip2", "zusammen"]

# Befundsatz (nach Praezisierung vom 17.10., Punkt 3; Lesart EO4-3)
BEFUNDFORMEN = ["show", "shows", "showed", "shown", "find", "finds", "found", "demonstrate", "demonstrates",
                "demonstrated", "observe", "observes", "observed", "reveal", "reveals", "revealed", "indicate",
                "indicates", "indicated", "suggest", "suggests", "suggested", "conclude", "concludes", "concluded",
                "confirm", "confirms", "confirmed", "prove", "proves", "proved", "proven", "result", "results"]
_RE_BEFUND = re.compile(r"\b(?:" + "|".join(BEFUNDFORMEN) + r")\b", re.IGNORECASE)

# Vermerk (nach Praezisierung vom 17.10., Punkt 4; Lesart EO4-2)
_RE_VERMERK_ANFANG = re.compile(r"\s*(?:\(c\)|©|copyright\b)", re.IGNORECASE)
_RE_VERMERK_KURZ = re.compile(
    r"\ball rights reserved\b"
    r"|\bpublished by\b.*\b(?:elsevier|wiley|springer|press|society|institute|publishing)\b"
    r"|\belsevier\s+(?:b\.?\s?v|ltd|inc|science|masson|gmbh)\b"
    r"|\bopen access article\b"
    r"|\bcreative commons\b", re.IGNORECASE | re.DOTALL)

# Woerter fuer EO4 A2b (nach Praezisierung vom 17.10., Punkt 5; Lesart EO4-6)
_RE_A2_WORT = re.compile(r"[a-z]+(?:'[a-z]+)?")

# Phrasen (nach Praezisierung vom 17.10., Punkt 1; Lesarten PHR-2 bis PHR-4)
STOPP = frozenset(ENGLISH_STOP_WORDS)
_RE_ABSCHNITT = re.compile(r"[.,;:!?()\[\]{}\"/]+")
_RE_INHALTSWORT = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")
RAND_PHRASEN = "-'"

# DI4 A4 (nach Praezisierung vom 17.10., Punkt 9)
DI4_DDOF = 1
# EO3 A1 (nach Praezisierung vom 17.10., Punkt 2)
EO3_QUANTILE = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


class Abbruch(Exception):
    """Fehler mit einem Bezeichner als Grund (keine Datentexte)."""


# ---------------------------------------------------------------------------------------------------------------
# Hilfen
# ---------------------------------------------------------------------------------------------------------------
def _maxrss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _sauber(x):
    """Macht ein Objekt JSON-tauglich (NaN zu None, numpy zu Python)."""
    if isinstance(x, dict):
        return {str(k): _sauber(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_sauber(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_sauber(v) for v in x.tolist()]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return None if not math.isfinite(v) else v
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def schreibe_json(pfad, obj):
    tmp = Path(str(pfad) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_sauber(obj), f, ensure_ascii=False, indent=1)
    tmp.replace(pfad)


def lies_json(pfad):
    if not Path(pfad).exists():
        return None
    with open(pfad, encoding="utf-8") as f:
        return json.load(f)


def speichere_npz(pfad, **arrays):
    tmp = Path(str(pfad) + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(pfad)


def lade_npz(pfad):
    if not Path(pfad).exists():
        raise Abbruch("fehlt_" + Path(pfad).stem)
    with np.load(pfad, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def topics_aus(labels):
    if labels.size == 0 or labels.max() < 0:
        raise Abbruch("keine_topics")
    return np.arange(int(labels.max()) + 1, dtype=np.int64)


def zeilen_je_topic(labels, topics):
    ordnung = np.argsort(labels, kind="stable")
    sortiert = labels[ordnung]
    links = np.searchsorted(sortiert, topics, side="left")
    rechts = np.searchsorted(sortiert, topics, side="right")
    return [np.sort(ordnung[a:b]) for a, b in zip(links, rechts)]


def mittel_oder_nan(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else float("nan")


# ---------------------------------------------------------------------------------------------------------------
# Laden
# ---------------------------------------------------------------------------------------------------------------
def lade_df(zusatzspalten):
    """Ladevorschrift der Schnittstelle (Eingabe 1), woertlich; gelesen werden nur die benoetigten Spalten der
    Kopfzeile aus der Schnittstelle (Title, Year, Abstract und je Teil die Zusatzspalten; Lesart ALLG-2).
    Die Umbenennung der Ladevorschrift wirkt nur auf vorhandene Spalten: danach heisst 'Author Full Names'
    'Authors' und 'Times Cited, WoS Core' 'Times Cited'."""
    gewuenscht = set(zusatzspalten) | {"Title", "Year", "Abstract"}
    kopf = set(pd.read_csv(P_CSV, nrows=0).columns)
    fehlend = sorted(gewuenscht - kopf)
    if fehlend:
        raise Abbruch("spalten_fehlen_" + "_".join(c.replace(" ", "").replace(",", "") for c in fehlend))
    df = pd.read_csv(P_CSV, usecols=lambda c: c in gewuenscht)
    # --- Ladevorschrift, unveraendert ---
    df = df.rename(columns={"Publication Year": "Year", "Author Full Names": "Authors",
                            "Source Title": "Source title", "Times Cited, WoS Core": "Times Cited"})
    df["text"] = df["Title"].fillna("") + ". " + df["Abstract"].fillna("")
    df["text_clean"] = df["text"].apply(
        lambda t: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s\-]", " ", t.lower())).strip() if isinstance(t, str) else "")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df[(df["Year"] >= 2000) & (df["Year"] <= 2015)].copy()
    df = df[df["text_clean"].str.split().str.len() >= 10].reset_index(drop=True)
    # --- Ende der Ladevorschrift ---
    return df


class _Platzhalter(dict):
    """Ersatz fuer Klassen im Pickle, deren Module hier fehlen (nur labels und embeddings_sbert werden benutzt)."""

    def __init__(self, *a, **k):
        dict.__init__(self)

    def __setstate__(self, state):
        pass

    def append(self, x):
        pass

    def extend(self, x):
        pass


class _NachsichtigerUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            return _Platzhalter


def lade_modell():
    with open(P_PKL, "rb") as f:
        try:
            d = pickle.load(f)
        except Exception:
            f.seek(0)
            d = _NachsichtigerUnpickler(f).load()
    labels = np.asarray(d["labels"]).astype(np.int64).ravel()
    emb = np.asarray(d["embeddings_sbert"], dtype=np.float64)
    del d
    gc.collect()
    return labels, emb


def lade_basis():
    return lade_npz(ZW / "basis.npz")


def lade_phrasen(basis):
    p = lade_npz(ZW / "phrasen.npz")
    if str(p["kennung"]) != str(basis["kennung"]):
        raise Abbruch("phrasen_passen_nicht_zu_basis")
    return p


def lade_einbettungen():
    pfad = ZW / "einbettungen.npy"
    if not pfad.exists():
        raise Abbruch("fehlt_einbettungen")
    return np.load(pfad)


def csr_aus(indptr, idx, n, k, binaer=False):
    X = sp.csr_matrix((np.ones(len(idx), dtype=np.int64), idx.astype(np.int64), indptr.astype(np.int64)),
                      shape=(n, k))
    X.sum_duplicates()
    if binaer:
        X.data[:] = 1
    return X


def phrasen_csr(P, n):
    return csr_aus(P["indptr"], P["indices"], n, int(P["vokabular"]), binaer=True)


# ---------------------------------------------------------------------------------------------------------------
# Textbausteine
# ---------------------------------------------------------------------------------------------------------------
def zelle_eintraege(zelle):
    if not isinstance(zelle, str):
        return []
    teile = [e.strip() for e in zelle.split(";")]
    return [e for e in teile if e]


def phrasen_menge(text, cache):
    """Uni- und Bigramme innerhalb von Laeufen aus Inhaltswoertern (Lesarten PHR-1 bis PHR-5, nach Praezisierung
    vom 17.10., Punkt 1). Gibt die Menge als dict in der Reihenfolge des ersten Auftretens zurueck."""
    menge = {}
    for abschnitt in _RE_ABSCHNITT.split(text.lower()):
        vorher = None
        for roh in abschnitt.split():
            wort = cache.get(roh)
            if wort is None:
                s = roh.strip(RAND_PHRASEN)
                wort = s if (len(s) >= 2 and s not in STOPP and _RE_INHALTSWORT.fullmatch(s) is not None) else ""
                cache[roh] = wort
            if wort:
                menge[wort] = None
                if vorher is not None:
                    menge[vorher + " " + wort] = None
                vorher = wort
            else:
                vorher = None
    return menge


def ist_vermerk(s):
    """Verlagsvermerk (Lesart EO4-2, nach Praezisierung vom 17.10., Punkt 4)."""
    if _RE_VERMERK_ANFANG.match(s) is not None:
        return True
    return len(s.split()) <= VERMERK_MAX_WOERTER and _RE_VERMERK_KURZ.search(s) is not None


def ist_befund(s):
    """Befundsatz (Lesart EO4-3, nach Praezisierung vom 17.10., Punkt 3)."""
    return _RE_BEFUND.search(s) is not None


def lade_lexikon():
    """Hedge- und Booster-Einwortlisten und ein gemeinsames Verzeichnis der Mehrwortausdruecke nach erstem Wort,
    je erstes Wort nach Laenge absteigend (laengster Ausdruck zuerst)."""
    with open(P_LEX, encoding="utf-8") as f:
        lex = json.load(f)
    h1 = frozenset(w.lower() for w in lex["hedges_einwort"])
    b1 = frozenset(w.lower() for w in lex["booster_einwort"])
    mehr = {}
    for art, schluessel in (("h", "hedges_mehrwort"), ("b", "booster_mehrwort")):
        for w in lex[schluessel]:
            tup = tuple(w.lower().split())
            mehr.setdefault(tup[0], []).append((tup, art))
    for k in mehr:
        mehr[k].sort(key=lambda e: -len(e[0]))
    return h1, b1, mehr


def zaehle_a2(text, h1, b1, mehr):
    """Woerter, Hedges und Booster eines Textes (Lesarten EO4-6 und EO4-7, nach Praezisierung vom 17.10., Punkt 5)."""
    w = _RE_A2_WORT.findall(text.lower())
    n = len(w)
    i = 0
    hedges = 0
    booster = 0
    while i < n:
        treffer = None
        for tup, art in mehr.get(w[i], ()):
            L = len(tup)
            if i + L <= n and tuple(w[i:i + L]) == tup:
                treffer = (L, art)
                break
        if treffer is not None:
            if treffer[1] == "h":
                hedges += 1
            else:
                booster += 1
            i += treffer[0]
            continue
        if w[i] in h1:
            hedges += 1
        elif w[i] in b1:
            booster += 1
        i += 1
    return n, hedges, booster


# ---------------------------------------------------------------------------------------------------------------
# Teil basis
# ---------------------------------------------------------------------------------------------------------------
def feld_csr(serie, klein=False):
    ids = {}
    indptr = np.zeros(len(serie) + 1, dtype=np.int64)
    idx = []
    n_mit = 0
    n_mehrfach = 0
    for i, zelle in enumerate(serie):
        e = zelle_eintraege(zelle)
        if klein:
            e = [x.lower() for x in e]
        if e:
            n_mit += 1
        loc = [ids.setdefault(x, len(ids)) for x in e]
        if len(set(loc)) < len(loc):
            n_mehrfach += 1
        idx.extend(loc)
        indptr[i + 1] = len(idx)
    info = {"dokumente_mit_eintrag": n_mit, "eintraege_gesamt": len(idx), "verschiedene_eintraege": len(ids),
            "verschiedene_ohne_schreibung": len({k.casefold() for k in ids}),
            "dokumente_mit_mehrfachnennung": n_mehrfach}
    return indptr, np.asarray(idx, dtype=np.int64), len(ids), info


def norm_name(s):
    """Namensnormierung fuer DI1 (nach Praezisierung vom 17.10., Punkt 6): Randleerraum weg, klein, einfache
    Leerzeichen."""
    return " ".join(s.lower().split())


def autoren_csr(serie_aut, serie_orc):
    ids = {}
    indptr = np.zeros(len(serie_aut) + 1, dtype=np.int64)
    idx = []
    z = {"orcid_eintraege": 0, "orcid_eintraege_mit_schraegstrich": 0, "orcid_eintraege_name_passt": 0,
         "orcid_eintraege_an_position_des_namens": 0, "autorennennungen": 0, "autorennennungen_mit_orcid": 0,
         "dokumente_ohne_autor": 0}
    for i, (aut, orc) in enumerate(zip(serie_aut, serie_orc)):
        namen = [norm_name(x) for x in zelle_eintraege(aut)]
        namen = [x for x in namen if x]
        omap = {}
        enamen = []
        for e in zelle_eintraege(orc):
            z["orcid_eintraege"] += 1
            if "/" not in e:
                continue
            nm, oid = e.rsplit("/", 1)
            nm = norm_name(nm)
            oid = oid.strip()
            if not nm or not oid:
                continue
            z["orcid_eintraege_mit_schraegstrich"] += 1
            enamen.append(nm)
            omap[nm] = oid  # bei gleichem Namen gilt der letzte Eintrag
        nameset = set(namen)
        for j, nm in enumerate(enamen):
            if nm in nameset:
                z["orcid_eintraege_name_passt"] += 1
                if j < len(namen) and namen[j] == nm:
                    z["orcid_eintraege_an_position_des_namens"] += 1
        if not namen:
            z["dokumente_ohne_autor"] += 1
        for nm in namen:
            z["autorennennungen"] += 1
            if nm in omap:
                schluessel = "o\t" + omap[nm]
                z["autorennennungen_mit_orcid"] += 1
            else:
                schluessel = "n\t" + nm
            idx.append(ids.setdefault(schluessel, len(ids)))
        indptr[i + 1] = len(idx)
    z["autorenkennungen"] = len(ids)
    z["autorenkennungen_orcid"] = sum(1 for k in ids if k.startswith("o\t"))
    return indptr, np.asarray(idx, dtype=np.int64), len(ids), z


def eo4_dokumente(n, abstracts):
    """Je Dokument: EO4 A1 (unbereinigt, zur Information), A1b, A2 (unbereinigt) und A2b."""
    werte = np.load(P_WERTE).astype(np.float64).ravel()
    with np.load(P_META, allow_pickle=False) as meta:
        m_doc = np.asarray(meta["doc"]).astype(np.int64).ravel()
        m_nr = np.asarray(meta["nr"]).astype(np.int64).ravel()
    if len(m_doc) != len(werte) or len(m_nr) != len(werte):
        raise Abbruch("saetze_meta_laenge")
    saetze = [None] * n
    with open(P_SAETZE, encoding="utf-8") as f:
        for zeile in f:
            if not zeile.strip():
                continue
            o = json.loads(zeile)
            i = int(o["doc"])
            if i < 0 or i >= n or saetze[i] is not None:
                raise Abbruch("saetze_doc_ungueltig")
            saetze[i] = o["saetze"]
    if any(s is None for s in saetze):
        raise Abbruch("saetze_fehlen_fuer_dokumente")
    h1, b1, mehr = lade_lexikon()

    a1 = np.full(n, np.nan)
    a1b = np.full(n, np.nan)
    a2_h = np.zeros(n)
    a2_b = np.zeros(n)
    a2_w = np.zeros(n)
    a2b_h = np.zeros(n)
    a2b_b = np.zeros(n)
    a2b_w = np.zeros(n)
    z = {"stuecke": 0, "bewertete_stuecke": 0, "befundstuecke_unbereinigt": 0, "vermerkstuecke_alle": 0,
         "vermerkstuecke_bewertet": 0, "dokumente_mit_befund_unbereinigt": 0, "dokumente_ohne_befund_unbereinigt": 0,
         "rueckfall_unbereinigt_auf_vermerk": 0, "dokumente_mit_vermerk": 0, "dokumente_ohne_bewertetes_stueck": 0,
         "dokumente_mit_befund_bereinigt": 0, "dokumente_rueckfall_bereinigt": 0,
         "dokumente_ohne_wert_a1b": 0, "abstracts_stueckwoerter_ungleich": 0,
         "werte_nicht_endlich": int((~np.isfinite(werte)).sum()), "werte_min": float(np.nanmin(werte)),
         "werte_max": float(np.nanmax(werte))}
    verschiebungen = set()
    k = 0
    K = len(werte)
    for d in range(n):
        stuecke = saetze[d]
        r = 0
        v_alle = []
        bef_alle = []
        verm_alle = []
        rest = []
        hat_vermerk = False
        for s in stuecke:
            s = s if isinstance(s, str) else str(s)
            z["stuecke"] += 1
            verm = ist_vermerk(s)
            if verm:
                hat_vermerk = True
                z["vermerkstuecke_alle"] += 1
            else:
                rest.append(s)
            if len(s.split()) >= SATZ_MIN_WOERTER:
                if k >= K:
                    raise Abbruch("mehr_bewertete_stuecke_als_werte")
                if m_doc[k] != d:
                    raise Abbruch("saetze_meta_doc_abweichend")
                verschiebungen.add(int(m_nr[k]) - r)
                if len(verschiebungen) > 1:
                    raise Abbruch("saetze_meta_nr_abweichend")
                v_alle.append(werte[k])
                bef_alle.append(ist_befund(s))
                verm_alle.append(verm)
                if verm:
                    z["vermerkstuecke_bewertet"] += 1
                k += 1
                r += 1
        if hat_vermerk:
            z["dokumente_mit_vermerk"] += 1
        # A1 unbereinigt (nur zur Information)
        if v_alle:
            v = np.asarray(v_alle, dtype=np.float64)
            b = np.asarray(bef_alle, dtype=bool)
            z["befundstuecke_unbereinigt"] += int(b.sum())
            if b.any():
                a1[d] = -float(v[b].mean())
                z["dokumente_mit_befund_unbereinigt"] += 1
            else:
                a1[d] = -float(v[-1])
                z["dokumente_ohne_befund_unbereinigt"] += 1
                if verm_alle[-1]:
                    z["rueckfall_unbereinigt_auf_vermerk"] += 1
            # A1b: ohne Vermerke
            keep = ~np.asarray(verm_alle, dtype=bool)
            if keep.any():
                vk = v[keep]
                bk = b[keep]
                if bk.any():
                    a1b[d] = -float(vk[bk].mean())
                    z["dokumente_mit_befund_bereinigt"] += 1
                else:
                    a1b[d] = -float(vk[-1])
                    z["dokumente_rueckfall_bereinigt"] += 1
            else:
                z["dokumente_ohne_wert_a1b"] += 1
        else:
            z["dokumente_ohne_bewertetes_stueck"] += 1
            z["dokumente_ohne_befund_unbereinigt"] += 1
            z["dokumente_ohne_wert_a1b"] += 1
        # A2 unbereinigt und A2b
        ab = abstracts[d] if isinstance(abstracts[d], str) else ""
        a2_w[d], a2_h[d], a2_b[d] = zaehle_a2(ab, h1, b1, mehr)
        if hat_vermerk:
            a2b_w[d], a2b_h[d], a2b_b[d] = zaehle_a2(" ".join(rest), h1, b1, mehr)
        else:
            a2b_w[d] = a2_w[d]
            a2b_h[d] = a2_h[d]
            a2b_b[d] = a2_b[d]
            if len(_RE_A2_WORT.findall(" ".join(str(s) for s in stuecke).lower())) != a2_w[d]:
                z["abstracts_stueckwoerter_ungleich"] += 1
    if k != K:
        raise Abbruch("weniger_bewertete_stuecke_als_werte")
    z["bewertete_stuecke"] = k
    z["nr_verschiebung"] = (list(verschiebungen)[0] if verschiebungen else 0)

    def dichte(h, w):
        out = np.full(n, np.nan)
        ok = w >= EO4_MIN_WOERTER
        out[ok] = 1000.0 * h[ok] / w[ok]
        return out

    z["dokumente_a2b_ab_20_woertern"] = int((a2b_w >= EO4_MIN_WOERTER).sum())
    z["dokumente_a2_ab_20_woertern"] = int((a2_w >= EO4_MIN_WOERTER).sum())
    z["woerter_a2b_gesamt"] = int(a2b_w.sum())
    z["hedges_a2b_gesamt"] = int(a2b_h.sum())
    z["booster_a2b_gesamt"] = int(a2b_b.sum())
    arrays = {"eo4_a1": a1, "eo4_a1b": a1b,
              "eo4_a2_dichte": dichte(a2_h, a2_w), "eo4_a2_booster_dichte": dichte(a2_b, a2_w),
              "eo4_a2b_dichte": dichte(a2b_h, a2b_w), "eo4_a2b_booster_dichte": dichte(a2b_b, a2b_w),
              "eo4_a2b_woerter": a2b_w}
    return arrays, z


def teil_basis():
    df = lade_df(["Author Full Names", "Times Cited, WoS Core", "Keywords Plus", "WoS Categories", "Affiliations",
                  "ORCIDs"])
    n = len(df)
    labels, emb = lade_modell()
    if labels.shape[0] != n:
        raise Abbruch("labels_laenge_ungleich_df")
    if emb.ndim != 2 or emb.shape[0] != n:
        raise Abbruch("einbettungen_laenge_ungleich_df")
    if labels.min() < -1:
        raise Abbruch("labels_ungueltig")
    jahr_f = df["Year"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(jahr_f)) or np.any(jahr_f != np.round(jahr_f)):
        raise Abbruch("jahr_ungueltig")
    jahr = jahr_f.astype(np.int64)
    zit = pd.to_numeric(df["Times Cited"], errors="coerce").to_numpy(dtype=np.float64)
    textlaenge = df["text_clean"].str.split().str.len().to_numpy(dtype=np.int64)

    if not np.all(np.isfinite(emb)):
        raise Abbruch("einbettungen_nicht_endlich")
    normen = np.linalg.norm(emb, axis=1)
    if np.any(normen <= 0):
        raise Abbruch("einbettung_null")
    emb_n = emb / normen[:, None]
    np.save(ZW / "einbettungen.npy", emb_n)
    info_emb = {"dimension": int(emb.shape[1]), "norm_min": float(normen.min()), "norm_max": float(normen.max())}
    del emb, emb_n
    gc.collect()

    info = {"dokumente": n, "einbettungen": info_emb}
    felder = {}
    for kurz, spalte in [("kat", "WoS Categories"), ("kwp", "Keywords Plus"), ("aff", "Affiliations")]:
        indptr, idx, k, fi = feld_csr(df[spalte], klein=(kurz == "kwp"))
        felder[kurz + "_indptr"] = indptr
        felder[kurz + "_idx"] = idx
        felder[kurz + "_k"] = np.array(k)
        info["feld_" + kurz] = fi
    indptr, idx, k, ai = autoren_csr(df["Authors"], df["ORCIDs"])
    felder["aut_indptr"] = indptr
    felder["aut_idx"] = idx
    felder["aut_k"] = np.array(k)
    info["autoren"] = ai
    info["zitationen_fehlend"] = int((~np.isfinite(zit)).sum())

    abstracts = df["Abstract"].tolist()
    del df
    gc.collect()
    eo4_arr, eo4_info = eo4_dokumente(n, abstracts)
    info["eo4_dokumente"] = eo4_info
    del abstracts
    gc.collect()

    h = hashlib.sha256()
    h.update(np.int64(n).tobytes())
    h.update(labels.astype(np.int64).tobytes())
    h.update(jahr.tobytes())
    h.update(textlaenge.tobytes())
    kennung = h.hexdigest()[:16]

    topics = topics_aus(labels)
    groessen = np.array([(labels == t).sum() for t in topics])
    info["topics"] = int(len(topics))
    info["topicgroesse_min"] = int(groessen.min())
    info["topicgroesse_max"] = int(groessen.max())
    info["topics_ohne_dokument"] = int((groessen == 0).sum())
    info["rauschdokumente"] = int((labels == -1).sum())
    info["kennung"] = kennung
    speichere_npz(ZW / "basis.npz", kennung=np.array(kennung), n=np.array(n), labels=labels, jahr=jahr,
                  zitationen=zit, textlaenge=textlaenge, **felder, **eo4_arr)
    return "fertig", info


# ---------------------------------------------------------------------------------------------------------------
# Teil phrasen
# ---------------------------------------------------------------------------------------------------------------
def teil_phrasen():
    B = lade_basis()
    n = int(B["n"])
    df = lade_df([])
    if len(df) != n:
        raise Abbruch("zeilenzahl_abweichend")
    tl = df["text_clean"].str.split().str.len().to_numpy(dtype=np.int64)
    jr = df["Year"].to_numpy(dtype=np.float64).astype(np.int64)
    if not (np.array_equal(tl, B["textlaenge"]) and np.array_equal(jr, B["jahr"])):
        raise Abbruch("zeilen_abweichend")
    texte = df["text"].tolist()
    del df
    gc.collect()

    cache = {}
    pid = {}
    unigramm = []
    dok = []
    for t in texte:
        m = phrasen_menge(t, cache)
        loc = np.empty(len(m), dtype=np.int64)
        for j, p in enumerate(m):
            q = pid.get(p)
            if q is None:
                q = len(pid)
                pid[p] = q
                unigramm.append(" " not in p)
            loc[j] = q
        dok.append(loc)
    del texte, cache
    n_phrasen = len(pid)
    del pid
    gc.collect()
    alle = np.concatenate(dok) if dok else np.zeros(0, dtype=np.int64)
    dfreq = np.bincount(alle, minlength=n_phrasen)
    del alle
    behalten = (dfreq >= VOK_MIN_DF) & (dfreq <= VOK_MAX_ANTEIL * n)
    V = int(behalten.sum())
    neu = np.full(n_phrasen, -1, dtype=np.int64)
    neu[behalten] = np.arange(V, dtype=np.int64)
    indptr = np.zeros(n + 1, dtype=np.int64)
    teile = []
    for i, loc in enumerate(dok):
        v = neu[loc]
        v = np.sort(v[v >= 0])
        teile.append(v)
        indptr[i + 1] = indptr[i] + len(v)
    indices = np.concatenate(teile).astype(np.int32) if teile else np.zeros(0, dtype=np.int32)
    uni_alle = np.asarray(unigramm, dtype=bool)
    uni = uni_alle[behalten]
    dfv = dfreq[behalten]
    L = np.diff(indptr)
    info = {"phrasen_verschieden": n_phrasen, "unigramme_verschieden": int(uni_alle.sum()),
            "bigramme_verschieden": int((~uni_alle).sum()), "vokabular": V,
            "vokabular_unigramme": int(uni.sum()), "vokabular_bigramme": int((~uni).sum()),
            "max_df_grenze": float(VOK_MAX_ANTEIL * n), "dokumente_ohne_vokabularphrase": int((L == 0).sum()),
            "vokabularphrasen_je_dokument_mittel": float(L.mean()) if n else float("nan"),
            "vokabular_eintraege_gesamt": int(L.sum())}
    speichere_npz(ZW / "phrasen.npz", kennung=B["kennung"], indptr=indptr, indices=indices, unigramm=uni, df=dfv,
                  vokabular=np.array(V))
    return "fertig", info


# ---------------------------------------------------------------------------------------------------------------
# Teil q2: EO1 A1, EO2 A1, IP3 A1, DI2 A1
# ---------------------------------------------------------------------------------------------------------------
def lambda_q2(X, zeilen):
    """Simpson-Konzentration ueber Paare von Eintraegen aus verschiedenen Dokumenten (1b)."""
    Xt = X[zeilen]
    L = np.asarray(Xt.sum(axis=1)).ravel().astype(np.int64)
    nk = np.asarray(Xt.sum(axis=0)).ravel().astype(np.int64)
    dat = Xt.data.astype(np.int64)
    N = int(L.sum())
    zaehler = int(np.dot(nk, nk)) - int(np.dot(dat, dat))
    nenner = N * N - int(np.dot(L, L))
    if nenner <= 0:
        return float("nan"), int((L > 0).sum()), N
    return zaehler / nenner, int((L > 0).sum()), N


def teil_q2():
    B = lade_basis()
    P = lade_phrasen(B)
    n = int(B["n"])
    labels = B["labels"]
    topics = topics_aus(labels)
    zeilen = zeilen_je_topic(labels, topics)
    felder = {
        "EO1_A1": phrasen_csr(P, n),
        "EO2_A1": csr_aus(B["kat_indptr"], B["kat_idx"], n, int(B["kat_k"])),
        "IP3_A1": csr_aus(B["kwp_indptr"], B["kwp_idx"], n, int(B["kwp_k"]), binaer=True),
        "DI2_A1": csr_aus(B["aff_indptr"], B["aff_idx"], n, int(B["aff_k"]), binaer=True),
    }
    erg = {}
    info = {}
    alle_topicdoks = np.concatenate(zeilen) if zeilen else np.zeros(0, dtype=np.int64)
    for name, X in felder.items():
        lam = np.full(len(topics), np.nan)
        ndok = np.zeros(len(topics), dtype=np.int64)
        neintr = np.zeros(len(topics), dtype=np.int64)
        for j, z in enumerate(zeilen):
            if len(z) == 0:
                continue
            lam[j], ndok[j], neintr[j] = lambda_q2(X, z)
        with np.errstate(divide="ignore", invalid="ignore"):
            if name == "DI2_A1":
                wert = lam.copy()
            else:
                wert = np.where(lam > 0, -np.log(np.where(lam > 0, lam, 1.0)), np.nan)
                if name == "EO2_A1":
                    wert = wert / math.log(N_KAT_NORM)
        erg[name] = wert
        erg[name + "_lambda"] = lam
        spalten_benutzt_korpus = int((np.diff(X.tocsc().indptr) > 0).sum())
        Xt = X[alle_topicdoks]
        spalten_benutzt_topics = int((np.diff(Xt.tocsc().indptr) > 0).sum())
        info[name] = {"topicdokumente_mit_eintrag": int(ndok.sum()), "eintraege_in_topics": int(neintr.sum()),
                      "verschiedene_eintraege_korpus": spalten_benutzt_korpus,
                      "verschiedene_eintraege_topicdokumente": spalten_benutzt_topics,
                      "topics_unter_10_dokumenten_mit_eintrag": int((ndok < 10).sum()),
                      "nan": int((~np.isfinite(wert)).sum())}
    info["EO2_A1"]["normierung_kategorien"] = N_KAT_NORM
    speichere_npz(ZW / "erg_q2.npz", kennung=B["kennung"], topics=topics, **erg)
    return "fertig", info


# ---------------------------------------------------------------------------------------------------------------
# Teil eo3: EO3 A1
# ---------------------------------------------------------------------------------------------------------------
def teil_eo3():
    B = lade_basis()
    E = lade_einbettungen()
    n = int(B["n"])
    labels = B["labels"]
    topics = topics_aus(labels)
    zeilen = zeilen_je_topic(labels, topics)
    laenge = B["textlaenge"].astype(np.float64)
    kanten = np.unique(np.quantile(laenge, EO3_QUANTILE))
    dez = np.searchsorted(kanten, laenge, side="right").astype(np.int64)
    K = len(kanten) + 1
    q = np.einsum("ij,ij->i", E, E)
    S = np.zeros((K, E.shape[1]))
    nn = np.zeros(K)
    Q = np.zeros(K)
    for a in range(K):
        sel = dez == a
        S[a] = E[sel].sum(axis=0)
        nn[a] = sel.sum()
        Q[a] = q[sel].sum()
    G = S @ S.T
    mu = G / np.outer(nn, nn)
    for a in range(K):
        mu[a, a] = (G[a, a] - Q[a]) / (nn[a] * (nn[a] - 1.0)) if nn[a] > 1 else np.nan
    St = E.sum(axis=0)
    korpus_mittel = (float(St @ St) - float(q.sum())) / (n * (n - 1.0))
    beob = np.full(len(topics), np.nan)
    erw = np.full(len(topics), np.nan)
    for j, z in enumerate(zeilen):
        nt = len(z)
        if nt < 2:
            continue
        s = E[z].sum(axis=0)
        beob[j] = (float(s @ s) - float(q[z].sum())) / (nt * (nt - 1.0))
        c = np.bincount(dez[z], minlength=K).astype(np.float64)
        P = np.outer(c, c)
        P[np.diag_indices(K)] = c * (c - 1.0)
        mu0 = np.where(P > 0, mu, 0.0)
        erw[j] = float((P * mu0).sum()) / (nt * (nt - 1.0))
    wert = erw - beob
    info = {"korpus_mittlere_kosinus_aller_paare": korpus_mittel, "klassen": K,
            "klassengrenzen": [float(x) for x in kanten], "dokumente_je_klasse": [int(x) for x in nn],
            "nan": int((~np.isfinite(wert)).sum()), "beobachtung_median": float(np.nanmedian(beob)),
            "erwartung_median": float(np.nanmedian(erw))}
    speichere_npz(ZW / "erg_eo3.npz", kennung=B["kennung"], topics=topics, EO3_A1=wert, beobachtung=beob,
                  erwartung=erw)
    return "fertig", info


# ---------------------------------------------------------------------------------------------------------------
# Teil eo4: EO4 A1b, A2b, R3b
# ---------------------------------------------------------------------------------------------------------------
def z_werte(x):
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    out = np.full(x.shape, np.nan)
    if ok.sum() >= 2:
        mu = x[ok].mean()
        sd = x[ok].std(ddof=0)
        if sd > 0:
            out[ok] = (x[ok] - mu) / sd
    return out


def teil_eo4():
    B = lade_basis()
    labels = B["labels"]
    topics = topics_aus(labels)
    zeilen = zeilen_je_topic(labels, topics)

    def topicmittel(arr):
        return np.array([mittel_oder_nan(arr[z]) for z in zeilen])

    def topicanzahl(arr):
        return np.array([int(np.isfinite(arr[z]).sum()) for z in zeilen])

    a1b = topicmittel(B["eo4_a1b"])
    a2b = topicmittel(B["eo4_a2b_dichte"])
    r3b = (z_werte(a1b) + z_werte(a2b)) / 2.0
    # zur Information (Vergleich mit 1b und 1g): unbereinigte Fassungen und Booster
    a1 = topicmittel(B["eo4_a1"])
    a2 = topicmittel(B["eo4_a2_dichte"])
    bo = topicmittel(B["eo4_a2_booster_dichte"])
    bob = topicmittel(B["eo4_a2b_booster_dichte"])

    def spearman(x, y):
        ok = np.isfinite(x) & np.isfinite(y)
        return float(stats.spearmanr(x[ok], y[ok])[0]) if ok.sum() > 2 else float("nan")

    info = {"dokumente_a1b": int(topicanzahl(B["eo4_a1b"]).sum()),
            "dokumente_a2b": int(topicanzahl(B["eo4_a2b_dichte"]).sum()),
            "topics_min_dokumente_a1b": int(topicanzahl(B["eo4_a1b"]).min()),
            "topics_min_dokumente_a2b": int(topicanzahl(B["eo4_a2b_dichte"]).min()),
            "woerter_a2b_topicdokumente": int(sum(w[w >= EO4_MIN_WOERTER].sum()
                                                  for w in (B["eo4_a2b_woerter"][z] for z in zeilen))),
            "nan_a1b": int((~np.isfinite(a1b)).sum()), "nan_a2b": int((~np.isfinite(a2b)).sum()),
            "nan_r3b": int((~np.isfinite(r3b)).sum()),
            "a2_unbereinigt_topics_median": float(np.nanmedian(a2)), "a2_unbereinigt_topics_min": float(np.nanmin(a2)),
            "a2_unbereinigt_topics_max": float(np.nanmax(a2)), "booster_unbereinigt_topics_median": float(np.nanmedian(bo)),
            "rho_hedges_booster_unbereinigt": spearman(a2, bo),
            "a2b_topics_median": float(np.nanmedian(a2b)), "booster_bereinigt_topics_median": float(np.nanmedian(bob)),
            "rho_a1b_a2b": spearman(a1b, a2b), "rho_a1_a1b": spearman(a1, a1b), "rho_a2_a2b": spearman(a2, a2b)}
    speichere_npz(ZW / "erg_eo4.npz", kennung=B["kennung"], topics=topics, EO4_A1B=a1b, EO4_A2B=a2b, EO4_R3B=r3b,
                  EO4_A1_unbereinigt=a1, EO4_A2_unbereinigt=a2)
    return "fertig", info


# ---------------------------------------------------------------------------------------------------------------
# Blockweise Teile (eo5, di1): fortsetzbar
# ---------------------------------------------------------------------------------------------------------------
def blockweise(teil, topics, kennung, parameter, rechne_block):
    """Rechnet Bloecke von Topics; vorhandene passende Bloecke werden uebernommen. Gibt (fertig, bloecke) zurueck."""
    bloecke = [topics[i:i + BLOCK_TOPICS] for i in range(0, len(topics), BLOCK_TOPICS)]
    ergebnisse = []
    neu_gerechnet = 0
    for b, tb in enumerate(bloecke):
        pfad = ZW / (teil + "_block_" + str(b).zfill(3) + ".npz")
        if pfad.exists():
            alt = lade_npz(pfad)
            if (str(alt["kennung"]) == kennung and str(alt["parameter"]) == parameter
                    and np.array_equal(alt["topics"], tb)):
                ergebnisse.append(alt)
                continue
        if neu_gerechnet >= 1 and time.perf_counter() - T_START > ZEITBUDGET_S:
            return False, len(ergebnisse), len(bloecke), None
        werte = rechne_block(tb)
        neu_gerechnet += 1
        speichere_npz(pfad, kennung=np.array(kennung), parameter=np.array(parameter), topics=tb, **werte)
        ergebnisse.append(lade_npz(pfad))
    zusammen = {}
    for k in ergebnisse[0]:
        if k in ("kennung", "parameter"):
            continue
        zusammen[k] = np.concatenate([e[k] for e in ergebnisse])
    return True, len(bloecke), len(bloecke), zusammen


# ---------------------------------------------------------------------------------------------------------------
# Teil eo5: EO5 A1
# ---------------------------------------------------------------------------------------------------------------
def teil_eo5():
    B = lade_basis()
    P = lade_phrasen(B)
    E = lade_einbettungen()
    n = int(B["n"])
    labels = B["labels"]
    topics = topics_aus(labels)
    zeilen = zeilen_je_topic(labels, topics)
    X = phrasen_csr(P, n)
    U = np.flatnonzero(P["unigramm"])
    Xu = X[:, U].tocsr()
    Xu.sort_indices()
    del X
    q = np.einsum("ij,ij->i", E, E)
    XuT = Xu.T.tocsr()
    n_w = np.diff(XuT.indptr).astype(np.int64)
    S_w = np.asarray(XuT @ E)
    Q_w = np.asarray(XuT @ q).ravel()
    del XuT

    def rechne_block(tb):
        wert = np.full(len(tb), np.nan)
        ndok = np.zeros(len(tb), dtype=np.int64)
        min_gueltig = np.zeros(len(tb), dtype=np.int64)
        nwort = np.zeros(len(tb), dtype=np.int64)
        ngueltig = np.zeros(len(tb), dtype=np.int64)
        for j, t in enumerate(tb):
            z = zeilen[int(t)]
            if len(z) == 0:
                continue
            Xt = Xu[z]
            if Xt.nnz == 0:
                continue
            cols, inv = np.unique(Xt.indices, return_inverse=True)
            inv = inv.ravel()
            Xc = sp.csr_matrix((np.ones(len(inv)), inv, Xt.indptr), shape=(len(z), len(cols)))
            n_wt = np.bincount(inv, minlength=len(cols)).astype(np.int64)
            XcT = Xc.T.tocsr()
            S_wt = np.asarray(XcT @ E[z])
            Q_wt = np.asarray(XcT @ q[z]).ravel()
            n_aus = n_w[cols] - n_wt
            D = S_w[cols] - S_wt
            sq = np.einsum("ij,ij->i", D, D)
            q_aus = Q_w[cols] - Q_wt
            gueltig = n_aus >= EO5_MIN_DOK
            cq = np.ones(len(cols))
            na = n_aus[gueltig].astype(np.float64)
            cq[gueltig] = (sq[gueltig] - q_aus[gueltig]) / (na * (na - 1.0))
            semd = np.where(gueltig, -np.log(np.maximum(cq, EO5_C_MIN)), 0.0)
            cnt = np.asarray(Xc @ gueltig.astype(np.float64)).ravel()
            tot = np.asarray(Xc @ semd).ravel()
            ok = cnt >= EO5_MIN_WOERTER
            dw = tot[ok] / cnt[ok]
            wert[j] = float(dw.mean()) if dw.size else np.nan
            ndok[j] = int(ok.sum())
            min_gueltig[j] = int(cnt.min())
            nwort[j] = len(cols)
            ngueltig[j] = int(gueltig.sum())
        return {"EO5_A1": wert, "dokumente": ndok, "min_gueltige_woerter": min_gueltig, "woerter": nwort,
                "gueltige_woerter": ngueltig}

    parameter = "m" + str(EO5_MIN_DOK) + "_w" + str(EO5_MIN_WOERTER) + "_c" + str(EO5_C_MIN)
    fertig, nb, gesamt, erg = blockweise("eo5", topics, str(B["kennung"]), parameter, rechne_block)
    if not fertig:
        return "fortsetzen", {"bloecke_fertig": nb, "bloecke_gesamt": gesamt}
    ok = n_w >= EO5_MIN_DOK
    with np.errstate(divide="ignore", invalid="ignore"):
        cq_voll = (np.einsum("ij,ij->i", S_w[ok], S_w[ok]) - Q_w[ok]) / (n_w[ok] * (n_w[ok] - 1.0))
    semd_voll = -np.log(np.maximum(cq_voll, EO5_C_MIN))
    rho = float(stats.spearmanr(semd_voll, n_w[ok])[0]) if ok.sum() > 2 else float("nan")
    gesamt_dok = int(sum(len(z) for z in zeilen))
    info = {"unigramme_vokabular": int(len(U)), "unigramme_ab_20_dokumenten_korpus": int(ok.sum()),
            "rho_semd_dokumentfrequenz_korpus": rho, "topicdokumente": gesamt_dok,
            "topicdokumente_mit_wert": int(erg["dokumente"].sum()),
            "topicdokumente_unter_5_gueltigen_woertern": gesamt_dok - int(erg["dokumente"].sum()),
            "min_gueltige_woerter_je_topicdokument": int(erg["min_gueltige_woerter"].min()),
            "nan": int((~np.isfinite(erg["EO5_A1"])).sum()), "bloecke": gesamt}
    speichere_npz(ZW / "erg_eo5.npz", kennung=B["kennung"], topics=topics, EO5_A1=erg["EO5_A1"])
    return "fertig", info


# ---------------------------------------------------------------------------------------------------------------
# Teil ds2: DS2 A3
# ---------------------------------------------------------------------------------------------------------------
def teil_ds2():
    B = lade_basis()
    P = lade_phrasen(B)
    n = int(B["n"])
    labels = B["labels"]
    jahr = B["jahr"]
    topics = topics_aus(labels)
    zeilen = zeilen_je_topic(labels, topics)
    X = phrasen_csr(P, n)
    Xc = X.tocsc()
    V = X.shape[1]
    nnz_sp = np.diff(Xc.indptr)
    erstjahr = np.full(V, np.iinfo(np.int64).max, dtype=np.int64)
    nz = nnz_sp > 0
    if nz.any():
        jr = jahr[Xc.indices]
        erstjahr[nz] = np.minimum.reduceat(jr, Xc.indptr[:-1][nz])
    L = np.diff(X.indptr)
    zeile = np.repeat(np.arange(n), L)
    neu = erstjahr[X.indices] >= (jahr[zeile] - DS2_FENSTER)
    zaehler = np.bincount(zeile, weights=neu.astype(np.float64), minlength=n)
    definiert = (jahr >= DS2_AB_JAHR) & (L > 0)
    anteil = np.full(n, np.nan)
    anteil[definiert] = zaehler[definiert] / L[definiert]
    dw = np.full(n, np.nan)
    jahresmittel = {}
    for y in np.unique(jahr[definiert]):
        sel = definiert & (jahr == y)
        mu = float(anteil[sel].mean())
        jahresmittel[int(y)] = mu
        dw[sel] = anteil[sel] - mu
    wert = np.array([mittel_oder_nan(dw[z]) for z in zeilen])
    ndok = np.array([int(np.isfinite(dw[z]).sum()) for z in zeilen])
    topicdoks = int(sum(len(z) for z in zeilen))
    info = {"dokumente_mit_wert_korpus": int(definiert.sum()), "anteil_korpus": float(definiert.mean()),
            "topicdokumente_mit_wert": int(ndok.sum()),
            "anteil_topicdokumente": float(ndok.sum() / topicdoks) if topicdoks else float("nan"),
            "jahrgangsmittel": jahresmittel, "nan": int((~np.isfinite(wert)).sum()),
            "topics_min_dokumente": int(ndok.min())}
    speichere_npz(ZW / "erg_ds2.npz", kennung=B["kennung"], topics=topics, DS2_A3=wert)
    return "fertig", info


# ---------------------------------------------------------------------------------------------------------------
# Teil di1: DI1 A1
# ---------------------------------------------------------------------------------------------------------------
def di1_topic(zeilen, aut_indptr, aut_idx, n_aut, m, R, rng):
    """R Ziehungen von m Dokumenten; je Ziehung Komponenten des Koautorennetzes (Lesart DI1-3).
    Gibt Werte, Komponentenzahlen und die gezogenen Positionen (Indizes in zeilen) zurueck."""
    n_t = len(zeilen)
    a = aut_indptr[zeilen]
    b = aut_indptr[zeilen + 1]
    laengen = (b - a).astype(np.int64)
    if laengen.sum() > 0:
        spalten = np.concatenate([aut_idx[x:y] for x, y in zip(a, b)])
    else:
        spalten = np.zeros(0, dtype=np.int64)
    zl = np.repeat(np.arange(n_t), laengen)
    Bt = sp.csr_matrix((np.ones(len(spalten)), (zl, spalten)), shape=(n_t, max(n_aut, 1)))
    M = (Bt @ Bt.T).toarray() > 0
    hat_autor = laengen > 0
    schluessel = rng.random((R, n_t))
    if m < n_t:
        idx = np.argpartition(schluessel, m - 1, axis=1)[:, :m]
    else:
        idx = np.argsort(schluessel, axis=1)[:, :m]
    sub = M[idx[:, :, None], idx[:, None, :]]
    bb, ii, jj = np.nonzero(sub)
    g = sp.csr_matrix((np.ones(len(bb), dtype=np.int8), (bb * m + ii, bb * m + jj)), shape=(R * m, R * m))
    _, lab = connected_components(g, directed=False)
    lab = lab.reshape(R, m).astype(np.int64)
    Lm = np.where(hat_autor[idx], lab, -1)
    Lm.sort(axis=1)
    neu = (Lm[:, 1:] != Lm[:, :-1]) & (Lm[:, 1:] >= 0)
    C = neu.sum(axis=1) + (Lm[:, 0] >= 0)
    werte = 1.0 - (C - 1.0) / (m - 1.0)
    return werte, C, idx


def di1_ziehungsmenge(zeilen, aut_indptr):
    """Topicdokumente mit mindestens einem Autor (nach Praezisierung vom 17.10., Punkt 6)."""
    return zeilen[(aut_indptr[zeilen + 1] - aut_indptr[zeilen]) > 0]


def teil_di1():
    B = lade_basis()
    labels = B["labels"]
    topics = topics_aus(labels)
    zeilen = zeilen_je_topic(labels, topics)
    aut_indptr = B["aut_indptr"]
    aut_idx = B["aut_idx"]
    n_aut = int(B["aut_k"])
    mengen = [di1_ziehungsmenge(z, aut_indptr) for z in zeilen]
    mit_autor = np.array([len(zm) for zm, z in zip(mengen, zeilen) if len(z) > 0])
    m = int(min(M_RAREF, mit_autor.min()))

    def rechne_block(tb):
        mw = np.full(len(tb), np.nan)
        sd = np.full(len(tb), np.nan)
        cmin = np.zeros(len(tb), dtype=np.int64)
        cmax = np.zeros(len(tb), dtype=np.int64)
        ohne = np.zeros(len(tb), dtype=np.int64)
        for j, t in enumerate(tb):
            z = zeilen[int(t)]
            zm = mengen[int(t)]
            ohne[j] = len(z) - len(zm)
            if len(zm) < m or m < 2:
                continue
            rng = np.random.default_rng([DI1_SEED, int(t)])
            w, C, _ = di1_topic(zm, aut_indptr, aut_idx, n_aut, m, DI1_ZIEHUNGEN, rng)
            mw[j] = float(w.mean())
            sd[j] = float(w.std(ddof=1)) if len(w) > 1 else np.nan
            cmin[j] = int(C.min())
            cmax[j] = int(C.max())
        return {"DI1_A1": mw, "DI1_A1_SD": sd, "komp_min": cmin, "komp_max": cmax, "ohne_autor": ohne}

    parameter = "m" + str(m) + "_r" + str(DI1_ZIEHUNGEN) + "_s" + str(DI1_SEED)
    fertig, nb, gesamt, erg = blockweise("di1", topics, str(B["kennung"]), parameter, rechne_block)
    if not fertig:
        return "fortsetzen", {"bloecke_fertig": nb, "bloecke_gesamt": gesamt}
    info = {"m": m, "ziehungen": DI1_ZIEHUNGEN, "seed": DI1_SEED, "komponenten_min": int(erg["komp_min"].min()),
            "komponenten_max": int(erg["komp_max"].max()), "topicdokumente_ohne_autor": int(erg["ohne_autor"].sum()),
            "sd_median": float(np.nanmedian(erg["DI1_A1_SD"])), "nan": int((~np.isfinite(erg["DI1_A1"])).sum()),
            "bloecke": gesamt}
    speichere_npz(ZW / "erg_di1.npz", kennung=B["kennung"], topics=topics, DI1_A1=erg["DI1_A1"],
                  DI1_A1_SD=erg["DI1_A1_SD"])
    return "fertig", info


# ---------------------------------------------------------------------------------------------------------------
# Teil di4_ip2: DI4 A4, IP2 A4
# ---------------------------------------------------------------------------------------------------------------
def zitationen_je_jahrgang(zit, jahr):
    """x (Zitationen durch Jahresmittel) und Perzentil je Jahrgang, beide ueber alle Korpusdokumente."""
    n = len(zit)
    gueltig = np.isfinite(zit)
    x = np.full(n, np.nan)
    pct = np.full(n, np.nan)
    for y in np.unique(jahr[gueltig]):
        sel = gueltig & (jahr == y)
        mu = float(zit[sel].mean())
        x[sel] = zit[sel] / mu if mu > 0 else 0.0
        pct[sel] = stats.rankdata(zit[sel], method="average") / float(sel.sum())
    return x, pct


def teil_di4_ip2():
    B = lade_basis()
    labels = B["labels"]
    jahr = B["jahr"]
    zit = B["zitationen"]
    topics = topics_aus(labels)
    zeilen = zeilen_je_topic(labels, topics)
    # IP2 A4 (unveraendert): Perzentile je Jahrgang ueber die Korpusdokumente mit Zitationszahl
    _, pct = zitationen_je_jahrgang(zit, jahr)
    # DI4 A4 (nach Praezisierung vom 17.10., Punkt 9): fehlende Zitationszahlen zaehlen als 0
    zit0 = np.where(np.isfinite(zit), zit, 0.0)
    x0, _ = zitationen_je_jahrgang(zit0, jahr)
    lx0 = np.log1p(x0)
    di4 = np.full(len(topics), np.nan)
    ip2 = np.full(len(topics), np.nan)
    ndok = np.zeros(len(topics), dtype=np.int64)
    ndok_di4 = 0
    for j, z_alle in enumerate(zeilen):
        if len(z_alle) >= DI4_MIN_DOK and float(x0[z_alle].sum()) > 0:
            sigma = float(lx0[z_alle].std(ddof=DI4_DDOF))
            di4[j] = 2.0 * float(stats.norm.cdf(sigma / math.sqrt(2.0))) - 1.0
            ndok_di4 += len(z_alle)
        z = z_alle[np.isfinite(zit[z_alle])]
        ndok[j] = len(z)
        if len(z) >= IP2_MIN_DOK and len(np.unique(jahr[z])) >= IP2_MIN_JAHRE:
            tau, _ = stats.kendalltau(jahr[z], pct[z])
            ip2[j] = float(tau) if np.isfinite(tau) else np.nan
    info = {"topicdokumente_mit_zitationen": int(ndok.sum()), "topicdokumente_di4": int(ndok_di4),
            "nan_di4": int((~np.isfinite(di4)).sum()), "nan_ip2": int((~np.isfinite(ip2)).sum()),
            "korpusdokumente_ohne_zitationen": int((~np.isfinite(zit)).sum())}
    speichere_npz(ZW / "erg_di4_ip2.npz", kennung=B["kennung"], topics=topics, DI4_A4=di4, IP2_A4=ip2)
    return "fertig", info


# ---------------------------------------------------------------------------------------------------------------
# Teil zusammen
# ---------------------------------------------------------------------------------------------------------------
ERGEBNISDATEIEN = {"erg_q2.npz": ["EO1_A1", "EO2_A1", "IP3_A1", "DI2_A1"], "erg_eo3.npz": ["EO3_A1"],
                   "erg_eo4.npz": ["EO4_A1B", "EO4_A2B", "EO4_R3B"], "erg_eo5.npz": ["EO5_A1"],
                   "erg_ds2.npz": ["DS2_A3"], "erg_di1.npz": ["DI1_A1", "DI1_A1_SD"],
                   "erg_di4_ip2.npz": ["DI4_A4", "IP2_A4"]}

PROTOKOLLWERTE = {
    "dokumente": 20706, "topics": 146, "bewertete_saetze": 109019, "phrasen_verschieden": 291928,
    "vokabular": 80858, "vokabular_unigramme": 19248, "eo5_unigramme_ab_20_dokumenten": 5100,
    "eo5_topicdokumente_unter_5_gueltigen_woertern": 0, "eo5_rho_semd_df": 0.10,
    "eo4_anteil_befundsaetze_unbereinigt": 0.240, "eo4_anteil_abstracts_mit_befund_unbereinigt": 0.732,
    "eo4_abstracts_ohne_befund_unbereinigt": 5557, "eo4_vermerkstuecke_bewertet": 2928,
    "eo4_rueckfall_auf_vermerk_unbereinigt": {"abschnitt_1g": 793, "aenderungseintrag_bereinigung": 796},
    "eo4_a2_topics_median": 5.6, "eo4_a2_topics_min": 2.4, "eo4_a2_topics_max": 15.8,
    "eo4_booster_topics_median": 10.1, "eo4_rho_hedges_booster": -0.01,
    "ds2_anteil_dokumente_mit_wert": 0.931, "di1_orcid_eintraege": 33951,
    "di1_orcid_eintraege_an_position_des_namens": 17744, "eo3_mittlere_kosinus_aller_paare": 0.318,
    "eo2_kategorien_korpus": 99,
}


def teil_zusammen():
    B = lade_basis()
    labels = B["labels"]
    topics = topics_aus(labels)
    kennung = str(B["kennung"])
    spalten = {}
    for datei, namen in ERGEBNISDATEIEN.items():
        e = lade_npz(ZW / datei)
        if str(e["kennung"]) != kennung:
            raise Abbruch("veraltet_" + datei.replace(".npz", ""))
        if not np.array_equal(e["topics"], topics):
            raise Abbruch("topics_abweichend_" + datei.replace(".npz", ""))
        for nm in namen:
            spalten[nm] = np.asarray(e[nm], dtype=np.float64)
    tab = pd.DataFrame({nm: spalten[nm] for nm in SPALTEN_AUSGABE}, index=pd.Index(topics, name="topic"))
    tab.to_csv(P_CSV_AUS, na_rep="NaN")

    laeufe = {}
    zaehl = {}
    for t in TEILE[:-1]:
        lj = lies_json(ZW / ("lauf_" + t + ".json"))
        if lj is None or lj.get("status") != "fertig":
            raise Abbruch("teil_nicht_fertig_" + t)
        if lj.get("kennung") not in (None, kennung) and t != "basis":
            raise Abbruch("teil_veraltet_" + t)
        laeufe[t] = {"sekunden": lj.get("sekunden_gesamt"), "aufrufe": len(lj.get("aufrufe", [])),
                     "speicher_mb_max": max(a.get("speicher_mb", 0) for a in lj.get("aufrufe", [{}]))}
        zaehl[t] = lj.get("info", {})

    gb = zaehl.get("basis", {})
    ge = gb.get("eo4_dokumente", {})
    gp = zaehl.get("phrasen", {})
    g5 = zaehl.get("eo5", {})
    g4 = zaehl.get("eo4", {})
    g2 = zaehl.get("ds2", {})
    g3 = zaehl.get("eo3", {})
    gq = zaehl.get("q2", {})
    ga = gb.get("autoren", {})
    n = int(B["n"])
    bew = ge.get("bewertete_stuecke", 0)
    zweit = {
        "dokumente": n, "topics": int(len(topics)), "bewertete_saetze": bew,
        "phrasen_verschieden": gp.get("phrasen_verschieden"), "vokabular": gp.get("vokabular"),
        "vokabular_unigramme": gp.get("vokabular_unigramme"),
        "eo5_unigramme_ab_20_dokumenten": g5.get("unigramme_ab_20_dokumenten_korpus"),
        "eo5_topicdokumente_unter_5_gueltigen_woertern": g5.get("topicdokumente_unter_5_gueltigen_woertern"),
        "eo5_rho_semd_df": g5.get("rho_semd_dokumentfrequenz_korpus"),
        "eo4_anteil_befundsaetze_unbereinigt": (ge.get("befundstuecke_unbereinigt", 0) / bew) if bew else None,
        "eo4_anteil_abstracts_mit_befund_unbereinigt": ge.get("dokumente_mit_befund_unbereinigt", 0) / n,
        "eo4_abstracts_ohne_befund_unbereinigt": ge.get("dokumente_ohne_befund_unbereinigt"),
        "eo4_vermerkstuecke_bewertet": ge.get("vermerkstuecke_bewertet"),
        "eo4_rueckfall_auf_vermerk_unbereinigt": ge.get("rueckfall_unbereinigt_auf_vermerk"),
        "eo4_a2_topics_median": g4.get("a2_unbereinigt_topics_median"),
        "eo4_a2_topics_min": g4.get("a2_unbereinigt_topics_min"),
        "eo4_a2_topics_max": g4.get("a2_unbereinigt_topics_max"),
        "eo4_booster_topics_median": g4.get("booster_unbereinigt_topics_median"),
        "eo4_rho_hedges_booster": g4.get("rho_hedges_booster_unbereinigt"),
        "ds2_anteil_dokumente_mit_wert": g2.get("anteil_topicdokumente"),
        "di1_orcid_eintraege": ga.get("orcid_eintraege"),
        "di1_orcid_eintraege_an_position_des_namens": ga.get("orcid_eintraege_an_position_des_namens"),
        "eo3_mittlere_kosinus_aller_paare": g3.get("korpus_mittlere_kosinus_aller_paare"),
        "eo2_kategorien_korpus": gq.get("EO2_A1", {}).get("verschiedene_eintraege_korpus"),
    }
    abgleich = {k: {"protokoll": PROTOKOLLWERTE[k], "zweitumsetzung": zweit.get(k)} for k in PROTOKOLLWERTE}
    nan_je_spalte = {nm: int((~np.isfinite(spalten[nm])).sum()) for nm in SPALTEN_AUSGABE}
    groessen = np.bincount(labels[labels >= 0], minlength=len(topics))
    g1 = zaehl.get("di1", {})
    g6 = zaehl.get("di4_ip2", {})
    je_fassung = {
        "alle": {"topicdokumente": int(groessen.sum()), "topicgroesse_min": int(groessen.min()),
                 "topicgroesse_max": int(groessen.max())},
        "EO1_A1": {"topicdokumente_mit_phrase": gq.get("EO1_A1", {}).get("topicdokumente_mit_eintrag"),
                   "phraseneintraege_in_topics": gq.get("EO1_A1", {}).get("eintraege_in_topics"),
                   "vokabular": gp.get("vokabular")},
        "EO2_A1": {"topicdokumente_mit_kategorie": gq.get("EO2_A1", {}).get("topicdokumente_mit_eintrag"),
                   "kategorieeintraege_in_topics": gq.get("EO2_A1", {}).get("eintraege_in_topics")},
        "EO3_A1": {"topicdokumente": int(groessen.sum()),
                   "paare_in_topics": int(sum(int(g) * (int(g) - 1) // 2 for g in groessen))},
        "EO4_A1B": {"dokumente_mit_wert": g4.get("dokumente_a1b"), "bewertete_saetze": bew},
        "EO4_A2B": {"dokumente_ab_20_woertern": g4.get("dokumente_a2b"),
                    "woerter_in_diesen_dokumenten": g4.get("woerter_a2b_topicdokumente")},
        "EO5_A1": {"topicdokumente_mit_wert": g5.get("topicdokumente_mit_wert"),
                   "unigramme_vokabular": g5.get("unigramme_vokabular")},
        "DS2_A3": {"topicdokumente_mit_wert": g2.get("topicdokumente_mit_wert")},
        "DI1_A1": {"m": g1.get("m"), "ziehungen": g1.get("ziehungen"),
                   "topicdokumente_ohne_autor": g1.get("topicdokumente_ohne_autor")},
        "DI2_A1": {"topicdokumente_mit_institution": gq.get("DI2_A1", {}).get("topicdokumente_mit_eintrag"),
                   "institutionseintraege_in_topics": gq.get("DI2_A1", {}).get("eintraege_in_topics")},
        "DI4_A4": {"topicdokumente_in_topics_mit_wert": g6.get("topicdokumente_di4"),
                   "korpusdokumente_ohne_zitationszahl_als_0": g6.get("korpusdokumente_ohne_zitationen")},
        "IP2_A4": {"topicdokumente_mit_zitationen": g6.get("topicdokumente_mit_zitationen")},
        "IP3_A1": {"topicdokumente_mit_keywords_plus": gq.get("IP3_A1", {}).get("topicdokumente_mit_eintrag"),
                   "keywords_plus_eintraege_in_topics": gq.get("IP3_A1", {}).get("eintraege_in_topics")},
    }
    info = {
        "skript": SKRIPT.name, "kennung_basis": kennung, "dokumente": n, "topics": int(len(topics)),
        "spalten": SPALTEN_AUSGABE, "nan_je_spalte": nan_je_spalte, "je_fassung": je_fassung,
        "parameter": {"m": M_RAREF, "di1_ziehungen": DI1_ZIEHUNGEN, "di1_seed": DI1_SEED,
                      "eo2_normierung_kategorien": N_KAT_NORM, "vokabular_min_df": VOK_MIN_DF,
                      "vokabular_max_anteil": VOK_MAX_ANTEIL, "eo5_min_dokumente": EO5_MIN_DOK,
                      "eo5_min_woerter": EO5_MIN_WOERTER, "eo5_c_min": EO5_C_MIN,
                      "eo4_min_woerter": EO4_MIN_WOERTER, "satz_min_woerter": SATZ_MIN_WOERTER,
                      "vermerk_max_woerter": VERMERK_MAX_WOERTER, "ds2_ab_jahr": DS2_AB_JAHR,
                      "ds2_fenster": DS2_FENSTER, "di4_min_dokumente": DI4_MIN_DOK,
                      "ip2_min_dokumente": IP2_MIN_DOK, "ip2_min_jahre": IP2_MIN_JAHRE},
        "abgleich_mit_protokollangaben": abgleich,
        "zaehlgroessen": zaehl,
        "laufzeit_je_teil": laeufe,
        "lesarten": LESARTEN,
        "versionen": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                      "scipy": __import__("scipy").__version__,
                      "sklearn": __import__("sklearn").__version__},
    }
    schreibe_json(P_INFO_AUS, info)
    return "fertig", {"nan_je_spalte": nan_je_spalte}


# ---------------------------------------------------------------------------------------------------------------
# Steuerung
# ---------------------------------------------------------------------------------------------------------------
FUNKTIONEN = {"basis": teil_basis, "phrasen": teil_phrasen, "q2": teil_q2, "eo3": teil_eo3, "eo4": teil_eo4,
              "eo5": teil_eo5, "ds2": teil_ds2, "di1": teil_di1, "di4_ip2": teil_di4_ip2,
              "zusammen": teil_zusammen}


def schreibe_lauf(teil, status, sekunden, info):
    pfad = ZW / ("lauf_" + teil + ".json")
    alt = lies_json(pfad) or {}
    kennung = None
    b = ZW / "basis.npz"
    if b.exists():
        try:
            with np.load(b, allow_pickle=False) as z:
                kennung = str(z["kennung"])
        except Exception:
            kennung = None
    aufruf = {"sekunden": round(sekunden, 3), "speicher_mb": round(_maxrss_mb(), 1), "status": status}
    if alt.get("status") == "fortsetzen" and alt.get("kennung") == kennung:
        aufrufe = alt.get("aufrufe", []) + [aufruf]
    else:
        aufrufe = [aufruf]
    d = {"teil": teil, "status": status, "kennung": kennung, "aufrufe": aufrufe,
         "sekunden_gesamt": round(sum(a["sekunden"] for a in aufrufe), 3), "info": info}
    schreibe_json(pfad, d)


def main(argv):
    if len(argv) != 2:
        print("aufruf=zweitumsetzung_phase1.py_teil")
        print("teile=" + ",".join(TEILE))
        return 2
    teil = argv[1]
    if teil == "liste":
        for t in TEILE:
            print(t)
        return 0
    if teil not in FUNKTIONEN:
        print("unbekannter_teil")
        print("teile=" + ",".join(TEILE))
        return 2
    ZW.mkdir(parents=True, exist_ok=True)
    debug = bool(os.environ.get("ZWEIT_DEBUG"))
    if not debug:
        warnings.filterwarnings("ignore")
    t0 = time.perf_counter()
    try:
        status, info = FUNKTIONEN[teil]()
    except Abbruch as e:
        print("teil=" + teil + " status=abbruch grund=" + str(e))
        return 1
    except Exception as e:
        if debug:
            traceback.print_exc()
        zeilen = [str(fr.lineno) for fr in traceback.extract_tb(e.__traceback__)]
        print("teil=" + teil + " status=fehler art=" + type(e).__name__ + " zeilen=" + ",".join(zeilen))
        return 1
    dt = time.perf_counter() - t0
    schreibe_lauf(teil, status, dt, info)
    print("teil=" + teil + " status=" + status + " sekunden=" + format(dt, ".1f") + " speicher_mb="
          + format(_maxrss_mb(), ".0f"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
