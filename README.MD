# Detekcija DDoS napada pomoću dubokih neuronskih mreža (DNN)

Projekat iz predmeta **Duboko učenje i neuronske mreže** — primena neuronskih mreža za predviđanje prevara u cyber bezbednosti (DDoS napadi).

## 1. Opis problema

DDoS (Distributed Denial of Service) napadi predstavljaju jednu od najčešćih i najdestruktivnijih pretnji u mrežnoj bezbednosti — napadač generiše ogroman broj zahteva/paketa ka cilju kako bi iscrpeo resurse servera ili mreže i onemogućio legitimne korisnike. Cilj ovog projekta je da se, na osnovu statističkih karakteristika mrežnog saobraćaja (tzv. *flow* feature-a — trajanje konekcije, broj i veličina paketa, vremenski intervali između paketa, TCP flag-ovi i sl.), izgradi model dubokog učenja (DNN) koji **binarno klasifikuje** svaki mrežni flow kao **benigni** ili **napad (DDoS)**.

Pristup je inspirisan radom *"Analysis of Deep and Graph Neural Networks for Enhanced DDoS Attack Detection"* (IEEE Access, 2025), koji poredi DNN i GNN pristupe za ovaj problem. Ovaj projekat se fokusira na DNN deo tog poređenja, uz sopstvenu analizu podataka, pipeline preprocesiranja i eksperimentalnu evaluaciju generalizacije modela na nezavisnom skupu podataka.

## 2. Podaci

### Izvor

- **Trening/test skup:** [InSDN Dataset (2020)](https://www.kaggle.com/datasets/muhammadumarjavaid/insdn-dataset-2020) — mrežni saobraćaj sniman u **Software-Defined Networking (SDN)** okruženju, sadrži i benigni i DDoS saobraćaj.
- **Nezavisni evaluacioni skup:** [CICIDS2017](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset) (Friday DDoS podskup) — mrežni saobraćaj sniman u tradicionalnoj (non-SDN) enterprise mreži, korišćen isključivo za testiranje generalizacije modela na potpuno nezavisnom skupu podataka.

Oba skupa su generisana korišćenjem **CICFlowMeter** alata, koji iz sirovog mrežnog saobraćaja (PCAP) ekstrahuje statističke feature-e po flow-u.

### Struktura

Originalni InSDN skup ima **343.889 redova i 84 kolone**:
- Identifikatori flow-a: `Flow ID`, `Src IP`, `Dst IP`, `Src Port`, `Dst Port`, `Protocol`, `Timestamp`
- Statistički feature-i: trajanje flow-a, broj/veličina paketa (forward/backward), IAT (inter-arrival time) statistike, TCP flag brojači, subflow statistike, aktivno/neaktivno vreme flow-a
- `target`: binarna labela (0 = benigni, 1 = napad)

Originalni balans klasa: **80.1% napad / 19.9% benigni**.

### Analiza i preprocesiranje

Preprocesiranje je urađeno kroz nekoliko koraka, od kojih je svaki motivisan konkretnim problemom uočenim tokom analize:

**a) Uklanjanje identifikatorskih kolona.** Kolone `Flow ID`, `Src IP`, `Dst IP` i `Timestamp` su izbačene. Enkodiranje IP adresa ili identifikatora flow-a kao numeričkih vrednosti (npr. `LabelEncoder`) uvodi lažan numerički poredak i predstavlja oblik *curenja podataka* (*data leakage*) — model bi mogao da nauči da prepoznaje specifične IP adrese iz trening skupa umesto stvarnih obrazaca saobraćaja, što ne bi generalizovalo na novi saobraćaj.

**b) Kategorizacija portova i protokola.** `Src Port`/`Dst Port` i `Protocol` su transformisani u kategorijalne vrednosti (npr. `80→http`, `443→https`, `53→dns`, `6→tcp`, `17→udp`, ostalo→`other`) na osnovu stvarne distribucije u podacima, a zatim one-hot enkodirani. Ovo čuva semantičku informaciju o tipu servisa/protokola bez veštačkog numeričkog poretka koji bi nastao enkodiranjem sirovih vrednosti porta.

**c) Uklanjanje konstantnih kolona.** Osam kolona (`Fwd Seg Size Min`, `Fwd/Bwd Byts/Pkts/b Avg`, `Fwd/Bwd Blk Rate Avg`, `Init Fwd Win Byts`) je imalo identičnu vrednost u svim redovima (poznat artefakt CICFlowMeter alata) — uklonjene su jer ne nose informacionu vrednost.

**d) Analiza i uklanjanje duplikata.** Nakon uklanjanja identifikatora, uočeno je da **120.517 redova (35%)** postaje identično u prostoru preostalih (bihevioralnih) feature-a, od čega **99.99% pripada klasi napada** — očekivana posledica prirode DDoS flood saobraćaja, gde se generiše veliki broj flow-ova sa praktično identičnim statističkim otiskom. Nakon dodatnog one-hot enkodiranja portova/protokola, pojavljuje se dodatnih ~38.000 duplikata iz istog razloga. Svi duplikati su uklonjeni **pre** podele na trening/test skup, kako bi se sprečilo da identičan obrazac saobraćaja završi i u treningu i u testu (što bi veštački naduvalo tačnost modela). Finalni dataset: **185.089 redova**, balans klasa **65.3% napad / 34.7% benigni** — realniji odraz raznovrsnosti saobraćaja.

**e) Transformacija raspodele (Yeo-Johnson).** Analizom je utvrđeno da su gotovo sve numeričke kolone ekstremno *skewed* (koeficijent asimetrije i preko 400 za pojedine kolone) — tipičan primer je `Tot Fwd Pkts`, gde je medijana 2, a maksimalna vrednost 910.748. Za DNN, ovakva raspodela otežava konvergenciju i čini standardno skaliranje (`StandardScaler`) nestabilnim (srednja vrednost i standardna devijacija su drastično iskrivljene retkim ekstremnim vrednostima). Primenjena je **Yeo-Johnson power transformacija** (generalizacija Box-Cox transformacije koja radi i sa nula i negativnim vrednostima), koja po koloni pronalazi optimalan parametar za normalizaciju raspodele, uz internu standardizaciju. Transformacija je **fitovana isključivo na trening skupu** i primenjena (`transform`) na test skup, kako bi se izbeglo curenje statistike test skupa u trening.

**f) Clipping.** Nakon transformacije, dodatno je primenjen clipping na opseg **[-5, 5]** kao sigurnosna mera protiv retkih preostalih ekstremnih vrednosti koje mogu destabilizovati trening.

**g) Podela na trening/test skup.** Stratifikovana podela 80/20 (148.071 trening / 37.018 test redova), čime je očuvan identičan balans klasa u oba skupa.

## 3. Arhitektura modela

Nakon eksperimentalnog poređenja šest arhitektura (videti sekciju 5), izabrana je najjednostavnija koja postiže vrhunske performanse:

```
Input(97 feature-a)
  → Dense(32, activation='relu')
  → Dropout(0.1)
  → Dense(16, activation='relu')
  → Dropout(0.1)
  → Dense(1, activation='sigmoid')
```

- **ReLU** u skrivenim slojevima — računski jeftina aktivacija koja ublažava problem *vanishing gradient* u odnosu na sigmoid/tanh, omogućavajući bolje propagiranje gradijenta kroz mrežu.
- **Sigmoid** u izlaznom sloju — mapira izlaz u opseg (0,1), tumačen kao verovatnoća pripadnosti klasi napada (odgovarajuće za binarnu klasifikaciju).
- **Dropout (0.1)** nakon svakog Dense sloja — regularizacija koja sprečava prekomerno prilagođavanje (*overfitting*) nasumičnim gašenjem dela neurona tokom treninga.
- **Binary crossentropy** kao funkcija gubitka — prirodno se uparuje sa sigmoid izlazom i daje informativne gradijente za klasifikacione probleme (za razliku od npr. MSE).
- **Adam optimizer**, learning rate = 0.001.

Model ima ukupno svega ~3.500 parametara — najmanji od svih testiranih varijanti, uz podjednako dobre (ili blago bolje) rezultate od znatno kompleksnijih arhitektura.

## 4. Trening

- **Batch size:** 64
- **Maksimalan broj epoha:** 50
- **EarlyStopping:** praćenje `val_loss`, `patience=5`, uz vraćanje najboljih težina (`restore_best_weights=True`) — trening automatski staje kada dalje epohe ne donose poboljšanje na validacionom skupu, sprečavajući overfitting i nepotrebno trošenje vremena.
- **Validacioni skup:** test skup korišćen je i kao validacioni tokom treninga (nema posebnog trećeg skupa s obzirom na veličinu dataseta i vremenska ograničenja projekta).

## 5. Analiza osetljivosti i hiperparametarska optimizacija

### Poređenje arhitektura

Testirano je šest konfiguracija (broj/veličina slojeva, dropout, learning rate), svaka trenirana do 15 epoha sa ranim zaustavljanjem (`patience=3`) radi brzog poređenja:

| Arhitektura | Slojevi | Dropout | LR | Val. accuracy | Val. loss |
|---|---|---|---|---|---|
| small | [32, 16] | 0.1 | 0.001 | 0.9997 | 0.0017 |
| medium | [64, 32] | 0.2 | 0.001 | 0.9998 | 0.0015 |
| large | [128, 64, 32] | 0.2 | 0.001 | 0.9998 | 0.0013 |
| wide | [256, 128] | 0.3 | 0.001 | 0.9998 | 0.0014 |
| original_like | [16, 256, 128] | 0.1 | 0.0001 | 0.9998 | 0.0016 |
| xtra-large | [256, 256, 256] | 0.05 | 0.0001 | 0.9998 | 0.0013 |

**Nalaz:** razlike u performansama između svih šest arhitektura su zanemarljive (u granicama šuma). Ovo ukazuje da problem, nakon pravilno sprovedenog preprocesiranja, postaje lako razdvojiv čak i za veoma jednostavne mreže — dodatni kapacitet ne donosi suštinsku korist. U skladu sa principom *Occam's razor*, izabrana je arhitektura **small** kao finalna.

### Poređenje sa/bez class_weight

S obzirom na blagu neizbalansiranost klasa (65/35), testirano je korišćenje `class_weight='balanced'` tokom treninga finalne (`small`) arhitekture. Rezultati (precision/recall/f1 po klasi) su bili **identični** sa i bez `class_weight` (do na dve decimale), pa je zadržan jednostavniji pristup bez dodatne komplikacije — neizbalansiranost od 65/35 nije dovoljno izražena da bi ovakva korekcija imala merljiv efekat.

## 6. Rezultati evaluacije

### Na InSDN test skupu (nezavisan test skup, isti izvor kao trening)

| Metrika | Vrednost |
|---|---|
| Accuracy | 99.97% |
| ROC AUC | 0.9999 |
| Precision (benigni) | 1.00 |
| Recall (benigni) | 1.00 |
| Precision (napad) | 1.00 |
| Recall (napad) | 1.00 |

Confusion matrix: od 37.018 test primera, model greši na svega **11** (8 lažno pozitivnih, 3 lažno negativna).

### Na CICIDS2017 (potpuno nezavisan skup, cross-dataset evaluacija)

Kada je isti model (treniran isključivo na InSDN podacima) primenjen na CICIDS2017 skup (bez ponovnog treninga), performanse značajno opadaju — F1 skor u opsegu **0.6–0.75**, u zavisnosti od tretmana feature-a portova/protokola (koji nisu identično dostupni u oba dataseta). Ovo je detaljnije razmotreno u sekciji Diskusija.

## 7. Diskusija

**Zašto su rezultati na InSDN skupu skoro savršeni?** Ovakva tačnost bi obično bila razlog za sumnju u curenje podataka. U ranim iteracijama pipeline-a, deo ove tačnosti je zaista bio veštački naduvan — enkodiranje IP adresa kao brojeva i neuklanjanje duplikata nastalih generalizacijom identifikatora omogućavali su modelu da delimično "pamti" specifične flow-ove iz trening skupa. Nakon sistematskog uklanjanja ovih izvora curenja (identifikatori, port/protokol generalizacija, duplikati), performanse ostaju vrlo visoke — ovo je sada konzistentno objašnjeno prirodom problema: DDoS flood saobraćaj proizvodi statistički vrlo izražene, lako uočljive obrasce (npr. drastično veći broj paketa po flow-u), što čini binarnu klasifikaciju suštinski lakim zadatkom kada su feature-i ispravno pripremljeni.

**Zašto performanse opadaju na CICIDS2017?** Pad F1 skora sa ~1.00 na 0.6–0.75 pri testiranju na nezavisnom skupu je poznat i dokumentovan fenomen u literaturi detekcije mrežnih napada — modeli trenirani na jednom skupu podataka često slabo generalizuju na drugi. Mogući doprinosni faktori:

- **Razlika u mrežnoj arhitekturi:** InSDN je sniman u **SDN (Software-Defined Networking)** okruženju, dok je CICIDS2017 sniman u tradicionalnoj mreži. Centralizovano upravljanje prosleđivanjem paketa u SDN-u (npr. inicijalna komunikacija sa kontrolerom pri uspostavljanju flow-a) može uticati na tajmning i statistike flow-a na način koji se razlikuje od tradicionalnog rutiranja.
- **Razlika u vremenu prikupljanja i alatima za generisanje napada** (InSDN: 2020, CICIDS2017: 2017) — različite verzije/implementacije DDoS alata mogu proizvesti blago drugačije statističke obrasce.
- **Razlike u ekstrakciji feature-a** — iako oba skupa koriste CICFlowMeter, moguće su verzijske razlike u implementaciji pojedinih metrika.

Nije moguće sa sigurnošću izolovati tačan doprinos svakog faktora bez dodatnih kontrolisanih eksperimenata, ali razlika u mrežnoj arhitekturi (SDN vs. non-SDN) je verovatan i naučno utemeljen doprinosni faktor, konzistentan sa poznatim izazovima cross-dataset generalizacije u ovoj oblasti.

**Ograničenja projekta:**
- Model je evaluiran na samo dva skupa podataka; šira evaluacija (više izvora, više tipova napada) bi dala pouzdaniju sliku generalizacije.
- Nije korišćena tehnika domenske adaptacije (*domain adaptation*) koja bi mogla ublažiti pad performansi pri prelasku između skupova.
- GNN pristup (koji je originalni referentni rad predložio kao superiorniji za detekciju low-volume/stealth napada zahvaljujući svesnosti o topologiji mreže) nije implementiran u ovom projektu zbog vremenskog ograničenja — ostaje kao pravac za proširenje rada.

## 8. Zaključak

Projekat je pokazao da DNN model, uz pažljivo i metodološki ispravno preprocesiranje (uklanjanje identifikatora, tretman duplikata nastalih iz prirode DDoS saobraćaja, normalizacija ekstremno asimetričnih raspodela), postiže gotovo savršenu detekciju DDoS napada (F1 ≈ 1.00, AUC = 0.9999) na test skupu istog porekla kao trening podaci. Analiza osetljivosti je pokazala da dodatni kapacitet mreže ne donosi merljivu korist na ovom problemu — jednostavna arhitektura (dva skrivena sloja, 32 i 16 neurona) postiže rezultate uporedive sa znatno kompleksnijim mrežama.

Evaluacija na nezavisnom CICIDS2017 skupu je otkrila značajan pad performansi (F1 0.6–0.75), ukazujući da model, iako izuzetno efikasan u prepoznavanju obrazaca specifičnih za trening domen (SDN saobraćaj), ne generalizuje podjednako dobro na saobraćaj iz drugačijeg mrežnog okruženja. Ovaj nalaz je vredan sam po sebi — naglašava važnost cross-dataset evaluacije pri procenjivanju IDS/DDoS modela i motiviše dalja istraživanja u pravcu robusnijih, domenski nezavisnih pristupa (npr. hibridni DNN-GNN modeli koji bolje hvataju topološke, a ne samo statističke karakteristike saobraćaja, ili tehnike domenske adaptacije).

---

## Reprodukcija

1. Instalirati zavisnosti: `pip install tensorflow scikit-learn pandas numpy matplotlib seaborn`
2. Preuzeti dataset-ove (Kaggle CLI kod dat u notebook-u)
3. Pokrenuti `DNN.ipynb` redom, ćeliju po ćeliju

## Licenca

Ovaj projekat je dostupan pod **MIT licencom** — videti `LICENSE` fajl.