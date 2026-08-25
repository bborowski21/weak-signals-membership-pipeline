# Changelog

Ältere Versionen sind über die Git-Tags `v1.2` bis `v2.2` dokumentiert.

## v2.3 (unveröffentlicht)

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

**Offen.** Die Konfidenzschwelle von 0,25 ist auf der c-TF-IDF-Skala kalibriert, auf der 95 Prozent
aller Topic-Paare einen Kosinus von exakt null tragen. Auf der SBERT-Skala liegt kein einziger
Best-Match mehr darunter, obwohl der schlechteste Treffer inhaltlich falsch ist. Die Schwelle ist neu
zu bestimmen, sinnvoll als empirisches Quantil statt als fester Wert. Bis dahin sind Aussagen über
„unsichere Matches“ nur innerhalb einer Cosine-Quelle vergleichbar.

**Hinweis zur Reproduzierbarkeit.** Tag `v2.2` bleibt unverändert der Stand, auf dem die Masterarbeit
beruht, einschließlich des dort verwendeten Aufrufs ohne `--with-sbert`. Wer die Zahlen der Arbeit
reproduzieren möchte, verwendet `v2.2` und startet `step01b` ohne das Flag. Wer methodenkonform zur
dokumentierten Definition rechnen möchte, verwendet `v2.3` mit `--with-sbert`.
