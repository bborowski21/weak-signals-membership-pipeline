# Changelog

Ältere Versionen sind über die Git-Tags `v1.2` bis `v2.2` dokumentiert.

## v2.4

Stand des Manuskripts. v2.4 behebt die Zählung der Reviews in DS3 (unten unter „Behoben“),
stellt die Glättung von DS3 auf den Review-Anteil der Phase um und hält die Laufumgebung samt
Revision des Sprachmodells fest. Der im Manuskript berichtete Lauf ist mit diesem Stand gerechnet
(vollständiger Neulauf vom 29./30.09.2026, Schritt 1 bis Schritt 6 samt Robustheitsexperimenten).
Seine Zahlen weichen von v2.3.1 ab, auch auf der Topic-Ebene (146 und 256 statt 146 und 265
Topics, 105 statt 101 gegenseitig beste Paare). Zahlen, die in den Einträgen unten beim Hinzufügen
einzelner Skripte genannt sind, stammen aus dem Lauf vom 21.05.2026.

### Geändert: Standard der Glättung von DS3

- `REVIEW_ABSENCE_PRIOR` in `config.py` steht jetzt auf `"phase_share"`: DS3 = 1 − (r + 2α·p₀)/(n + 2α)
  mit p₀ = Σr/Σn, dem Review-Anteil der Phase über alle Topics (im berichteten Lauf 2,04 und
  3,22 Prozent), bei unveränderter Stärke 2α = 10. Der symmetrische Prior (Mittelwert 0,5) zieht
  jedes Topic zur Hälfte, obwohl nur 2 bis 3 Prozent der Publikationen Reviews sind; DS3 bliebe
  damit auch nach der korrigierten Zählung an die Topicgröße gebunden (Spearman mit log n 0,90 und
  0,77 gegen 0,12 und −0,01 unter `"phase_share"`, Korrelation mit PE1 −0,75 und −0,63 gegen
  −0,16 und −0,01; Phase 1 und Phase 2). `"symmetric"` bleibt als Option erhalten.

### Hinzugefügt

- `paper_figures.py`: eigenständige Publikationsfassung aller Abbildungen als Kopie neben der
  Pipeline. Liest ausschließlich vorhandene Artefakte (output_phase1/2, output_cross_phase) und
  schreibt nach `figures_paper/` (gitignored) PNG 300 dpi plus Vektor-PDF. Designlinie wie das
  überarbeitete Sankey: englische Beschriftung (Dimensionsnamen provisorisch, finale Terminologie
  beim Kick-off), keine eingebetteten Titel, keine Em-Dashes, Low-n-Dämpfung in der
  Shift-Heatmap (Zellen mit n < 3 grau). Neu darin: `signature_scatter`, die drei
  Cross-Phase-Signaturen aus Thesis-Abschnitt 3.3 auf der Zerlegung (σ_sem, σ_lex) mit
  empirischen Terzilgrenzen (hoch: ≥ oberes Terzil, niedrig: strikt < unteres Terzil; auf den
  101 SBERT-Paaren: stabil 17, Concept Drift 4, semantisch instabil 34). Die Pipeline-Skripte
  selbst bleiben unverändert.


- **Abbildungssatz für die Zeitschrifteneinreichung**: `rp_style.py` (Stilmodul) und
  `paper_figures_rp.py` (22 Abbildungen plus drei Parkplatz-Abbildungen). Eigenständige Fassung
  neben `paper_figures.py`; die Pipeline-Skripte bleiben unverändert. Zielbreiten 90 / 140 / 190 mm,
  Export je Abbildung als Vektor-PDF mit eingebetteter Schrift, TIFF (RGB, LZW, 600 dpi), PNG
  (Sichtprobe, 300 dpi) und Graustufenprobe, dazu `manifest.csv` und `palette.json`. Vier
  Klassenfarben mit Zweitkodierung über Marker, Linientyp und Schraffur, damit Farbe nie
  alleiniger Informationsträger ist; Phasenvergleiche monochrom. Schreibt nach `figures_rp/`
  (gitignored).
- **Robustheitsexperimente** als eigenständige Skripte, alle drei lesen ausschließlich abgeleitete
  Artefakte aus `output_phase1/` und `output_phase2/` und rechnen mit dem unveränderten
  Pipeline-Code (`step02_indicators.compute_dimension_scores`, `step02_memberships.compute_memberships`):
  - `perturbation_experiment.py`: additives Normalrauschen je Indikator mit Standardabweichung
    s · SD_j, Stufen s = 2 / 5 / 10 / 20 Prozent, 1.000 Replikate je Stufe und Phase, Seed 20260903.
    Kennzahlen: Kipprate des Argmax gesamt, je Margin-Klasse und je Baseline-Konfiguration,
    Übergangsmatrix, Kippwahrscheinlichkeit und häufigste Ausweichklasse je Topic, mittlere
    absolute Membership-Verschiebung, Spearman je Membership. Bei 10 Prozent Rauschen kippt die
    dominante Konfiguration in 6,9 Prozent (Phase 1) und 7,3 Prozent (Phase 2) der
    Topic-Replikat-Paare; kein Topic mit Margin ≥ 0,10 kippt in Phase 1.
  - `nullmodel_experiment.py`: zwei Nullmodelle mit exakt erhaltener Randverteilung jedes
    Indikators. N1 permutiert jede der 16 Indikatorspalten unabhängig, N2 verschiebt alle
    Indikatoren einer Dimension mit derselben Permutation. 1.000 Replikate, Seed 20260904.
    Kennzahlen unter anderem mittlere Within-Dimension-Korrelation (real 0,227 und 0,223 gegen
    0,066 und 0,049 permutiert) und die Korrelation zwischen Weak-Signal- und
    Emerging-Concept-Membership.
  - `standardisation_variants.py`: Vergleich dreier Standardisierungsvarianten der 16 Indikatoren
    (Code-Referenz, robust-z vor der Dimensionsaggregation, robust-z ohne zweite Standardisierung).
    Die Argmax-Übereinstimmung mit der Referenz liegt bei 86,3 und 84,2 Prozent.
- `build_supplement.py`: erzeugt die Supplement-Tabellen S1 bis S4 (Indikatorkorrelationen je
  Phase, Dimensionsscores, Memberships, Indikatorwerte) nach `supplement_rp/` (gitignored).
  Enthält ausschließlich abgeleitete Größen, keine Rohdaten und keine bibliographischen Angaben.

- **Schalter für den Prior von DS3** (`REVIEW_ABSENCE_PRIOR` in `config.py`, Parameter `prior` in
  `compute_review_absence`): `"phase_share"` (Standard ab v2.4, siehe oben) oder `"symmetric"`
  (Mittelwert 0,5, DS3 = 1 − (r + α)/(n + 2α), die Formel bis v2.3.1). Das α-Gitter der
  Sensitivitätsanalyse wirkt in beiden Fällen als Stärke 2α.
- **Revision des Sprachmodells festgeschrieben**: `SBERT_MODEL_REVISION` in `config.py`, der Commit
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` von all-MiniLM-L6-v2 auf dem Hugging Face Hub
  (01.06.2026); `compute_embeddings` in `step01_topic_modeling.py` lädt das Modell in dieser
  Revision. Gegenüber der Vorgängerrevision `c9745ed1` (06.03.2025) ist nur die Modellkarte
  geändert, Gewichte, Konfiguration und Tokenizer sind gleich.
- `step06_cross_phase_transitions.py` (Schritt 6): Übergänge der dominanten Konfiguration über die
  gegenseitig besten Paare, als CSV und JSON nach `output_cross_phase/`.
- `run_phase_boundary_stability.py`: verschiebt die Phasengrenze über 2014 bis 2020 und vergleicht
  jede Aufteilung gegen einen unter denselben Einstellungen neu gerechneten Referenzschnitt
  (Supplement-Tabelle S6 und Abbildung S16); `make_figS16.py` zeichnet die Abbildung aus den
  Ausgabedateien, ohne hart eingetragene Zahlen.
- `pruefe_reproduzierbarkeit.py`: hält HDBSCAN und UMAP getrennt gegen die gespeicherten Ausgaben
  eines Laufs und zeigt so, in welchem Schritt ein Neulauf abweicht.
- `requirements.lock.txt`: die 64 Distributionen der Laufumgebung des berichteten Laufs
  (Python 3.12.7, macOS arm64), einschließlich factor_analyzer 0.5.1.
- `analyse_konfidenz.py` sowie `RELEASE_v2.3.md` und `RELEASE_v2.3.1.md`: Nachrechnung der drei
  Kippbedingungen zur Match-Konfidenz aus `topic_matches_full.csv` und die Release-Notizen der
  beiden Vorgängerversionen.
- `CITATION.cff`: Zitationsangaben (Titel, Autor mit ORCID, Lizenz, Version) für GitHub und Zenodo.

### Geändert

- `plot_migration_sankey` in `step04b_cross_phase_viz.py` durch eine Publikationsfassung ersetzt.
  Vorher: zwei Bänder je Klassenpaar (klar + schraffiert „knapp“), Deckkraft je Band nach
  Match-Cosine, quellseitige Stapelung nach Bandgröße. Bei 73 von 101 knappen Migrationen
  dominierte die Schraffur, das Cosine-Encoding ist auf der SBERT-Skala uninformativ, und die
  Stapelung erzeugte Kreuzungen direkt am Knoten. Jetzt: ein aggregiertes Band je Klassenpaar,
  Stapelung quellseitig nach Zielklasse und zielseitig nach Quellklasse (minimale Kreuzungen),
  einheitliche Deckkraft mit weißen Trennkanten, n an Knoten und an Bändern ab n = 3 (beide
  Enden), englische Beschriftung ohne eingebetteten Titel, zusätzlich Vektor-PDF
  (`migration_sankey.pdf`). Die klar/knapp-Zählung (Margin < 0.10) bleibt in stdout-Bericht und
  Rückgabewert erhalten. Matching und Zahlen unverändert; die Migrationsmatrix wurde unabhängig
  gegen `topic_matches_mutual.csv` und `signal_memberships.csv` verifiziert (Zeilensummen
  30/32/12/27, Spaltensummen 23/10/18/50, Diagonale 38/101).

- `.gitignore`: `figures_rp/` und `supplement_rp/` ergänzt, damit die generierten Abbildungs- und
  Supplement-Artefakte lokal bleiben; dazu alle Ausgabeordner über `output_*/*` und alle
  `*.npy`-Dateien. Einzige Ausnahme ist `output_phase_boundary/phase_boundary_stability.json`.
- `build_supplement.py`: Die Datendateien heißen jetzt wie im Supplement des Manuskripts,
  `indicator_correlations_phase1.csv` und `_phase2.csv`, `topic_dimension_scores.csv`,
  `topic_memberships.csv` und `topic_indicators.csv`, ohne die Vorsilben S1 bis S4; die S-Nummern
  bezeichnen im Supplement Tabellen und Abbildungen. Der Inhalt ist unverändert (byte-gleich
  gegen die Ausgabe der Vorgängerfassung geprüft).
- `paper_figures_rp.py`: FigA6 und FigA7 nehmen die Spaltenbreiten, FigA9 die Topiczahlen der
  Legende aus den Daten statt der fest eingetragenen 146 und 265. FigA9 wird auf 140 × 110 mm
  gebaut, damit die Achsenbeschriftung vollständig steht. In Abbildung 1 steht der Trend beim
  Wirkungspotenzial bei 3,5 statt 2,7, über der Skalenmitte und unter Emerging Concept.
- `pruefe_reproduzierbarkeit.py`: Optionen `--dir` (Ordner mit `model_results.pkl`) und `--out`;
  reproduzieren beide Stufen exakt, ist das Urteil jetzt neutral formuliert statt auf eine
  Abweichung des Stabilitätslaufs bezogen.
- Belegdateien auf den berichteten Lauf gezogen: `output_phase_boundary/phase_boundary_stability.json`,
  `reproduzierbarkeit_phase1.json`, `reproduzierbarkeit_phase2.json` und `topic_quality_results.json`.
- `README.md`: Zitation, Konfiguration, Installation und Versionierung auf v2.4.

### Behoben

- **DS3 (Review-Absenz, in der Thesis EP3) zählte keine Reviews.** `compute_review_absence` in
  `step02_indicators.py` verglich `Document Type` mit `"Review"`; die aufbereiteten KATI-Daten
  führen die Typen klein (`review`). Die Zählung war in allen Topics null und der Indikator damit
  eine reine Funktion der Topicgröße, 1 − α/(n_t + 2α). Der Vergleich ist jetzt unabhängig von der
  Schreibweise und erkennt mehrteilige Angaben (`Review; Early Access` im WoS-Export,
  `early access article|review` bei KATI). Im Analysekorpus stehen 418 (Phase 1) und 1.351
  (Phase 2) Reviews, davon 289 und 861 in Topics. Betroffen sind alle vorliegenden Ausgaben ab
  Schritt 2, auch die des Laufs vom 21.05.2026; der im Manuskript berichtete Lauf ist mit v2.4
  neu gerechnet.
- `clean_doctype` in `prepare_kati_data.py` übernahm den ersten Eintrag der KATI-Typenliste; bei
  `early access article|review` ging der Typ Review verloren (Phase 2: 35 Records der Lieferung,
  davon 10 im Analysekorpus). „early access article“ gilt jetzt als Status, nicht als Typ.
  Wirksam erst nach erneuter Aufbereitung mit `python prepare_kati_data.py --force`.
- `step03_efa_pca.py` läuft mit scikit-learn ab 1.8. factor_analyzer 0.5.1 ruft
  `check_array(force_all_finite=...)`; scikit-learn hat den Parameter in 1.6 umbenannt und in 1.8
  entfernt. Ein Aufsatz übersetzt das Schlüsselwort, wenn der alte Name fehlt. Muster, Phi und
  Kommunalitäten sind damit bitgleich mit scikit-learn 1.5 ohne Aufsatz. factor_analyzer stand in
  `requirements.txt`, fehlte aber in der Laufumgebung und in `requirements.lock.txt`.

## v2.3.1

Patch auf v2.3. Eine erneute Prüfung hat gezeigt, dass v2.3 die Abweichung dokumentiert, sie in
zwei Punkten aber selbst neu aufstellte. Dieser Patch schließt die Lücken und präzisiert drei
Formulierungen des v2.3-Eintrags (unten im v2.3-Abschnitt als „Präzisierung v2.3.1“ markiert).

### Behoben

- `run_all_phases.py` ruft `step01b` jetzt mit `--with-sbert` auf. Zuvor erzeugte der vollständige
  Orchestrator-Lauf weiterhin c-TF-IDF-Matches und legte sie neben die SBERT-basierte
  Sensitivitätsdatei aus `step05`, also exakt die Mischlage, die v2.3 als Ursache beschreibt.
- Die `topic_matches`-CSVs und `sensitivity_hybrid_alpha.csv` tragen eine neue Spalte
  `cosine_source` (`sbert_centroid` oder `ctfidf`). Die Provenienz steht damit in den Artefakten
  selbst statt nur in einer Textzeile des Diagnostikberichts, und Ausgaben der beiden Betriebsmodi
  bleiben auch nach dem Lauf unterscheidbar. Beide repo-internen Konsumenten
  (`step04b_cross_phase_viz.py`, `generate_ws_topic_tables.py`) greifen spaltenbasiert zu und sind
  von der zusätzlichen Spalte unberührt.
- Der Diagnostikbericht verwendet kein Konfidenz-Vokabular mehr: „Score-Schwelle (deskriptiv)“
  statt „Konfidenzschwelle“, „Matches unter Score-Schwelle (Review-Kandidaten)“ statt „Unsichere
  Matches“. Begründung im v2.3-Eintrag: Der Hybrid-Score misst Ähnlichkeit, nicht Korrektheit.
- Der Fallback-Hinweis bei angefordertem, aber nicht ladbarem SBERT ist von `[info]` auf
  `[WARNUNG]` angehoben und nennt die resultierende `cosine_source`.
- README: Die dort gelistete Konstante `HYBRID_ALPHA` existierte in `config.py` nicht (das
  Cosine-Gewicht liegt als `DEFAULT_ALPHA` in `step01b`); korrigiert. Neuer Abschnitt „Zwei
  Betriebsmodi des Cross-Phase-Matchings“ erklärt beide Modi und ihre Artefakt-Kennzeichnung.

### Bewusst nicht enthalten (wartet auf das Kick-off mit den Ko-Autoren)

Kanonischer Default bzw. Pflichtparameter für `use_sbert` (eine Frage der künftigen
Repo-Ausrichtung, ehrlich ein v2.4- oder v3.0-Schritt mit Ankündigung), die Frage, ob eine
Review-Triage-Liste je Match ins Paper gehört, die englische Paper-Terminologie sowie die
Umstellung der Visualisierungs-Konsumenten auf einen getrennten SBERT-Ausgabeordner.

## v2.3

### Behoben: Cross-Phase-Matching lief auf einer anderen Cosine-Quelle als dokumentiert

**Sachverhalt.** `step01b_cross_phase_matching.py` stellt die semantische Komponente des
Hybrid-Scores über das Opt-in-Flag `--with-sbert` ein. Ohne dieses Flag verwendet das Skript den
Kosinus der c-TF-IDF-Keyword-Vektoren statt des Kosinus der SBERT-Topic-Zentroide. Die
Sensitivitätsanalyse in `step05_sensitivity.py` ruft dieselbe Funktion über
`hybrid_alpha_sensitivity_cross_phase(..., use_sbert: bool = True)` auf und verließ sich auf diesen
Default. Derselbe Parameter hatte damit bei zwei Aufrufern zwei verschiedene Werte.

Die dem Thesis-Stand (Tag `v2.2`) zugrunde liegenden Läufe des Cross-Phase-Matchings wurden ohne
`--with-sbert` gestartet. Sie verwenden daher c-TF-IDF, während die zugehörige Methodendokumentation
die semantische Ähnlichkeit als Kosinus der SBERT-Zentroide definiert. Die Sensitivitätsanalyse folgt
der Definition, der Hauptlauf nicht.

*Präzisierung v2.3.1:* Die Masterarbeit ist an dieser Stelle intern uneinheitlich, nicht einseitig
falsch. Kapitel 3 (Gl. 3.1/3.2) definiert die semantische Ähnlichkeit über SBERT-Zentroide, der
Implementierungsanhang D beschreibt das Matching dagegen ausdrücklich „auf Basis der
c-TF-IDF-Repräsentationen“ und dokumentiert damit korrekt, was tatsächlich lief.

**Auswirkung.** Beide Varianten wurden auf identischer Datengrundlage nachgerechnet:

| Kenngröße | c-TF-IDF | SBERT-Zentroid |
|---|---|---|
| Mutual-Best-Paare | 99 | 101 |
| Schnittmenge beider Mengen | 82 | 82 |
| mean Hybrid (mutual) | 0,3624 | 0,6590 |
| stärkstes Paar | 0,758 | 0,852 |
| Best-Matches unter Schwelle 0,25 | 63 von 146 | 0 von 146 |
| Korrelation der beiden Score-Komponenten | +0,957 | +0,426 |

Die letzte Zeile ist der methodisch erhebliche Punkt: Unter c-TF-IDF messen semantische und
lexikalische Komponente des Hybrid-Scores nahezu dasselbe, weil beide auf denselben Top-15-Keywords
beruhen. Die Gewichtung der beiden Komponenten verliert damit ihre Bedeutung.

Nicht betroffen sind alle phasenintern berechneten Größen: Topic-Modelle, die 16 Indikatoren,
robust-z-Transformation, Memberships, EFA, externe Validierung, OAT-Sensitivität und
Zitations-Kohärenz. Diese Schritte greifen nicht auf das Cross-Phase-Matching zu.

**Änderungen.**

- `step01b_cross_phase_matching.py`: Der Diagnostikbericht unterscheidet nun drei Fälle statt zwei.
  Bisher lautete das Label in jedem Nicht-SBERT-Fall „c-TF-IDF (Fallback)“, unabhängig davon, ob
  SBERT angefordert und das Laden fehlgeschlagen war oder ob SBERT nie angefordert wurde. Genau diese
  fehlende Unterscheidung machte die Abweichung über mehrere Läufe hinweg unauffällig.
- `step01b_cross_phase_matching.py`: Läuft das Skript ohne SBERT, weist es auf stdout ausdrücklich
  darauf hin, dass die semantische Komponente nicht der Methodendefinition entspricht.
- `step05_sensitivity.py`: Der Aufruf setzt `use_sbert=True` explizit, statt sich auf den Default zu
  verlassen. Kommentar am Aufrufort erklärt, warum.
- `.gitignore`: Ergänzt um `output_cross_phase*/`, damit auch abweichend benannte Ausgabeordner
  ausgeschlossen bleiben. Zuvor griffen nur `*.csv` und `*.log`, sodass etwa
  `match_diagnostics.txt` aus einem solchen Ordner versehentlich eingecheckt werden konnte.

**Konfidenzschwelle: ersatzlos gestrichen statt neu kalibriert.** Die Schwelle von 0,25 war auf der
c-TF-IDF-Skala kalibriert, auf der 95 Prozent aller Topic-Paare einen Kosinus von exakt null tragen.
Sie auf der SBERT-Skala neu zu bestimmen setzt voraus, dass ein niedriger Score auf einen falschen
Match hindeutet. Das trifft nicht zu:

- Unter den zehn schwächsten Zeilenmaxima auf der SBERT-Skala stehen fachlich korrekte Zuordnungen,
  darunter P1#6 (berry phase) auf P2#9 (holonomic, nonadiabatic holonomic), P1#32 (hidden subgroup
  problem) auf P2#135 (quantum query, boolean functions) und P1#41 (entanglement concentration) auf
  P2#231 (entanglement witnesses). Ohne Fachurteil nachvollziehbar ist P1#103 (mutually unbiased,
  specker, kochen) auf P2#212 (contextuality, contextual, ks): „ks“ steht für Kochen-Specker.
- Der Margin, also der Abstand vom besten zum zweitbesten Treffer, liefert ebenfalls keinen Beleg
  für eine Trennung: Spearman gegen das Zeilenmaximum liegt bei 0,783, und die Wertebereiche der als
  richtig und als falsch beurteilten Paare überlappen vollständig. *Präzisierung v2.3.1:* Bei der
  geringen Zahl beurteilter Fälle ist das Fehlen eines Belegs die korrekte Lesart, nicht der Nachweis
  des Gegenteils.

*Präzisierung v2.3.1 zur Herkunft der Beurteilungen:* Die Richtig/Falsch-Einschätzungen der
schwächsten Matches stammen aus einer unverblindeten Einzeldurchsicht (KI-gestützt, n = 10) und sind
kein Ersatz für eine fachliche Doppelkodierung. Für die zentrale Aussage genügt bereits ein einzelner
belegbar korrekter Match am unteren Ende; der eindeutigste Fall (P1#103 „mutually unbiased, specker,
kochen“ auf P2#212 „contextuality, ks“, wobei „ks“ für Kochen-Specker steht) ist aus den
Schlüsselwörtern selbst ablesbar. Die Einzelbewertung von P1#24 (fullerene auf Dy/single-molecule
magnets) ist dagegen unsicher, da Dy-endohedrale Metallofullerene eine etablierte SMM-Forschungslinie
sind.

Der Hybrid-Score misst Ähnlichkeit, nicht Korrektheit. Ein Maß je Match, das Konfidenz behauptet,
ist deshalb nicht belegbar, unabhängig von seiner Berechnung. Die Diagnostik weist die Zahl unsicherer
Matches weiterhin aus, aber als deskriptive Angabe zur gewählten Schwelle, nicht als Gütemaß.

**Stattdessen: Auswertung auf der Zerlegung.** Die drei Cross-Phase-Signaturen sind in der
Methodendokumentation auf dem Wertepaar aus semantischer und lexikalischer Ähnlichkeit definiert,
nicht auf dem Hybrid-Score. Auf der c-TF-IDF-Skala ist die Concept-Drift-Zelle bei Terzilteilung
leer (0 von 99 Mutual-Paaren), auf der SBERT-Skala besetzt (4 von 101; bei Mediansplit 3 gegen 7).
Die Signaturschicht wird erst auf dem methodenkonformen Repräsentationsraum auswertbar.

**Hinweis zur Reproduzierbarkeit.** Tag `v2.2` bleibt unverändert der Stand, auf dem die Masterarbeit
beruht, einschließlich des dort verwendeten Aufrufs ohne `--with-sbert`. Wer die Zahlen der Arbeit
reproduzieren möchte, verwendet `v2.2` und startet `step01b` ohne das Flag. Wer methodenkonform zur
dokumentierten Definition rechnen möchte, verwendet `v2.3` mit `--with-sbert`.
