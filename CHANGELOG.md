# Changelog

Ältere Versionen sind über die Git-Tags `v1.2` bis `v2.2` dokumentiert.

## Unveröffentlicht

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
