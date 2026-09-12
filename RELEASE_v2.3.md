## v2.3

Korrigiert eine stille Abweichung zwischen Methodendokumentation und tatsächlichem Lauf und schließt
die Frage nach einem Konfidenzmaß für Cross-Phase-Matches ab.

### Die Ursache

Der Hybridscore des Cross-Phase-Matchings soll die semantische Ähnlichkeit aus SBERT-Topic-Zentroiden
bilden. `step01b_cross_phase_matching.py` stellt das über das Opt-in-Flag `--with-sbert` ein, dessen
Default `False` ist. `step05_sensitivity.py` ruft dieselbe Funktion über `use_sbert: bool = True` auf
und verlässt sich auf diesen Default. Ein Parameter, zwei Aufrufstellen, gegenläufige Vorgaben.

Die dem Thesis-Stand (`v2.2`) zugrunde liegenden Läufe rechneten daher auf den c-TF-IDF-Keyword-Vektoren.
`match_diagnostics.txt` protokollierte das durchgehend als „Cosine-Quelle: c-TF-IDF (Fallback)", aber
das Label lautete auch dann so, wenn SBERT nie angefordert wurde.

### Auswirkung

Beide Varianten auf identischer Datengrundlage nachgerechnet:

| Kenngröße | c-TF-IDF | SBERT-Zentroid |
|---|---|---|
| Mutual-Best-Paare | 99 | 101 |
| davon in beiden Mengen | 82 | 82 |
| stärkstes Paar | 0,758 | 0,852 |
| Rangkorrelation der Score-Komponenten | 0,9675 | 0,4508 |
| Concept-Drift-Zelle (Terzilteilung) | 0 von 99 | 4 von 101 |
| beidphasig WS-dominante Paare | 10 von 99 | 13 von 101 |

Die vorletzte Zeile ist der methodisch erhebliche Punkt. Unter c-TF-IDF messen semantische und
lexikalische Komponente auf den berichteten Paaren nahezu dasselbe, die Gewichtung bleibt wirkungslos,
und die Signaturtypologie ist empirisch nicht besetzbar.

**Nicht betroffen** sind alle phasenintern berechneten Größen: Topic-Modelle, die 16 Indikatoren,
robust-z, Memberships, Margin, EFA, externe Validierung, OAT-Sensitivität und Zitations-Kohärenz.

### Konfidenzschwelle

Die Schwelle von 0,25 wurde **nicht neu kalibriert, sondern als Gütemaß gestrichen**. Ihre
Neubestimmung würde voraussetzen, dass niedrige Scores auf falsche Matches hindeuten. Das trifft nicht
zu: Unter den zehn schwächsten Treffern stehen fachlich korrekte Zuordnungen, und der Margin trennt
ebenfalls nicht. Der Hybridscore misst Ähnlichkeit, nicht Korrektheit. Ausgewertet wird stattdessen
auf der Zerlegung, auf der die drei Cross-Phase-Signaturen ohnehin definiert sind.

### Änderungen

- `step01b`: Diagnostik unterscheidet „SBERT nicht angefordert" von „angefordert und fehlgeschlagen"
- `step01b`: Warnhinweis auf stdout bei Läufen ohne SBERT
- `step05_sensitivity`: `use_sbert=True` explizit am Aufrufort
- `.gitignore`: `output_cross_phase*/` ergänzt
- `CHANGELOG.md` neu

### Kompatibilität

`v2.2` bleibt unverändert der reproduzierbare Stand der Masterarbeit, einschließlich des dort
verwendeten Aufrufs ohne `--with-sbert`. Wer die Zahlen der Arbeit reproduzieren möchte, verwendet
`v2.2`. Wer methodenkonform zur dokumentierten Definition rechnen möchte, verwendet `v2.3` mit
`--with-sbert`.
