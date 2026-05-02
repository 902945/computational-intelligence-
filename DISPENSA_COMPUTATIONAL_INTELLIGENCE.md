# DISPENSA DI COMPUTATIONAL INTELLIGENCE
### Basata sui seminari di Ca' Foscari / Milano-Bicocca

**Fonti:**
1. *Investigating Robustness in Complex Systems: The Role of Model's Parameters* — Prof. Daniela Besozzi (Università di Milano-Bicocca)
2. *Supervised and Semi-Supervised Explainable AI applied to Cardiovascular Magnetic Resonance Mapping Techniques* — Matteo Grazioso (Ca' Foscari University of Venice)
3. *Drug Design in the era of Machine Learning and Computational Intelligence* — Silvia Multari (Ca' Foscari University of Venice)

---

## INDICE

1. [Robustezza nei Sistemi Complessi](#1-robustezza-nei-sistemi-complessi)
   1. [Che cos'è la robustezza?](#11-che-cosè-la-robustezza)
   2. [Meccanismi della robustezza](#12-meccanismi-della-robustezza)
   3. [Robustezza in diversi contesti scientifici](#13-robustezza-in-diversi-contesti-scientifici)
   4. [Trade-off e fragilità: il cancro come sistema robusto](#14-trade-off-e-fragilità-il-cancro-come-sistema-robusto)
   5. [Risposta robusta: restare o cambiare?](#15-risposta-robusta-restare-o-cambiare)
   6. [Quando la robustezza viene persa: bistabilità e punti critici](#16-quando-la-robustezza-viene-persa-bistabilità-e-punti-critici)
2. [Robustezza e Parametri del Modello](#2-robustezza-e-parametri-del-modello)
   1. [Parameter Estimation](#21-parameter-estimation)
   2. [Parameter Sweep Analysis (PSA)](#22-parameter-sweep-analysis-psa)
   3. [Sensitivity Analysis (SA)](#23-sensitivity-analysis-sa)
   4. [Bifurcation Theory](#24-bifurcation-theory)
3. [Caso Studio: Pathway Ras/cAMP/PKA nel Lievito](#3-caso-studio-pathway-rascamp-pka-nel-lievito)
4. [AI e Deep Learning per la Diagnosi Cardiovascolare (CMR)](#4-ai-e-deep-learning-per-la-diagnosi-cardiovascolare-cmr)
   1. [Background medico: MRI e CMR](#41-background-medico-mri-e-cmr)
   2. [Il paradigma quantitativo: T1 e T2 mapping](#42-il-paradigma-quantitativo-t1-e-t2-mapping)
   3. [Barriere all'adozione clinica](#43-barriere-alladozione-clinica)
   4. [Apprendimento supervisionato](#44-apprendimento-supervisionato)
   5. [Loss function: Focal Loss e Label Smoothing](#45-loss-function-focal-loss-e-label-smoothing)
   6. [Apprendimento semi-supervisionato (SSL)](#46-apprendimento-semi-supervisionato-ssl)
   7. [Model Ensembling](#47-model-ensembling)
   8. [Explainability (XAI)](#48-explainability-xai)
   9. [Fairness nell'AI clinica](#49-fairness-nellai-clinica)
   10. [Dettagli tecnici: ottimizzazione degli iperparametri e fine-tuning](#410-dettagli-tecnici-ottimizzazione-degli-iperparametri-e-fine-tuning)
5. [Drug Design nell'Era del Machine Learning e della Computational Intelligence](#5-drug-design-nellera-del-machine-learning-e-della-computational-intelligence)
   1. [Cos'è un farmaco e come si sviluppa](#51-cosè-un-farmaco-e-come-si-sviluppa)
   2. [Lo spazio chimico e la sfida computazionale](#52-lo-spazio-chimico-e-la-sfida-computazionale)
   3. [Approcci al Drug Design computazionale](#53-approcci-al-drug-design-computazionale)
   4. [Predire le reazioni metaboliche con un Molecular Transformer](#54-predire-le-reazioni-metaboliche-con-un-molecular-transformer)
   5. [Ottimizzazione di peptidi ciclici con Monte Carlo e Molecular Dynamics](#55-ottimizzazione-di-peptidi-ciclici-con-monte-carlo-e-molecular-dynamics)
   6. [Evoluzione di molecole con un Algoritmo Genetico](#56-evoluzione-di-molecole-con-un-algoritmo-genetico)
6. [Connessioni Trasversali e Concetti Chiave per l'Esame](#6-connessioni-trasversali-e-concetti-chiave-per-lesame)

---

## 1. Robustezza nei Sistemi Complessi

### 1.1 Che cos'è la robustezza?

La **robustezza** è una proprietà fondamentale che permette a un sistema complesso di **mantenere la propria funzionalità nonostante perturbazioni esterne o interne**. Si tratta di un concetto centrale in ingegneria, biologia, ecologia e informatica, e la sua comprensione richiede di abbandonare l'idea intuitiva (ma sbagliata) che un sistema robusto sia quello che "non cambia mai" davanti agli stimoli.

Un esempio classico è il sistema di controllo automatico del volo (AFCS, Automatic Flight Control System) degli aerei: esso mantiene direzione, altitudine e velocità nonostante le turbolenze atmosferiche. Ma attenzione: se l'aereo subisce una perturbazione inusuale come un'interruzione totale di corrente, questo sistema non basta più. Il punto cruciale è che **la robustezza non è illimitata né assoluta**, ma è sempre relativa a una classe specifica di perturbazioni.

> **Attenzione all'equivoco frequente:** robustezza non significa immutabilità. Un sistema robusto *cambia* il proprio modo di operare in modo flessibile per *preservare* le funzioni essenziali. La struttura interna e i componenti possono trasformarsi, purché le funzionalità critiche vengano mantenute.

La robustezza è, per definizione, una **proprietà a livello di sistema**: non può essere compresa guardando i singoli componenti isolatamente. Solo dall'interazione tra le parti emerge la capacità del sistema di resistere alle perturbazioni.

---

### 1.2 Meccanismi della robustezza

Quali sono gli strumenti che la natura (e l'ingegneria) usano per rendere robusti i sistemi? Se ne individuano principalmente tre:

**1. Feedback regulation (regolazione a retroazione)**
La retroazione è il meccanismo per cui l'output di un sistema influenza l'input successivo. Nei sistemi biologici, i feedback negativi tendono a stabilizzare: se una molecola viene prodotta in eccesso, essa stessa inibisce la propria produzione. I feedback positivi, al contrario, amplificano i segnali e possono portare a transizioni di stato (ad esempio, all'accensione di un gene). La combinazione di loop di feedback positivi e negativi è alla base di comportamenti complessi come oscillazioni e bistabilità.

**2. Fail-safe mechanisms: ridondanza e diversità**
Un sistema diventa più robusto se possiede componenti ridondanti (più unità capaci di svolgere la stessa funzione) progettate in modo diverso l'una dall'altra. Il caso dell'AFCS ne è l'esempio: il sistema è composto da 3 moduli con le stesse funzioni ma progettati diversamente, così da evitare il "common mode failure" (il fallimento contemporaneo di tutti i moduli per la stessa causa). In biologia, la ridondanza genica ha un ruolo analogo.

**3. Modularità**
Un sistema modulare è suddiviso in sottosistemi relativamente indipendenti (moduli) che svolgono funzioni specifiche e si interfacciano con il resto del sistema attraverso interfacce ben definite. Se un modulo viene danneggiato, il suo guasto tende a restare confinato, senza propagarsi all'intero sistema. Anche il genoma degli organismi viventi è organizzato in moduli funzionali.

---

### 1.3 Robustezza in diversi contesti scientifici

Il termine "robustezza" compare in molte discipline scientifiche, e il suo significato cambia a seconda del tipo di perturbazione che il sistema deve tollerare:

- **Scienze dei materiali:** un materiale è robusto se, rimossa la forza deformante, recupera la propria struttura e forma originale. Può esibire isteresi transitoria o lento rilassamento, ma lo stato iniziale viene infine ripristinato.

- **Sistemi ecologici:** l'introduzione di una specie aliena può inizialmente destabilizzare l'ecosistema, provocando la proliferazione della nuova specie e la scomparsa di alcune specie native. Con il tempo, però, il sistema si adatta e l'ecosistema originario può essere quasi ripristinato. La robustezza ecologica è dunque una capacità di assorbimento e riequilibrio dinamico.

- **Sistemi biologici:** in biologia, la robustezza è intrecciata con la concetto di **adattabilità** (evolution). I meccanismi di feedback, eterogeneità e ridondanza operano a scale diverse (molecolare, cellulare, tissutale, organismica). Un ruolo fondamentale è svolto dal **rumore biologico**: fluttuazioni stocastiche nell'espressione genica e nelle concentrazioni molecolari che, paradossalmente, possono contribuire alla robustezza del sistema permettendo l'esplorazione di diversi stati funzionali.

---

### 1.4 Trade-off e fragilità: il cancro come sistema robusto

Uno dei risultati più controintuitivi ma importanti della teoria della robustezza è che **essa non è gratuita**: l'introduzione di meccanismi di feedback genera inevitabilmente dei trade-off.

Il prezzo della robustezza è duplice:
- **Instabilità:** perturbazioni inaspettate possono innescare fallimenti catastrofici in un sistema altrimenti ben funzionante.
- **Riduzione delle prestazioni** in condizioni normali.

L'esempio più potente di questo trade-off è il **cancro visto come sistema robusto** (Kitano, Nature Reviews Cancer, 2004). I meccanismi che normalmente proteggono l'organismo vengono "dirottati" e riconfigurati per sostenere e promuovere lo stato patologico. In altre parole, il cancro sfrutta le stesse strategie di robustezza dell'organismo sano (feedback, ridondanza, modularità) per garantire la propria sopravvivenza nonostante le terapie.

Questa visione ha importanti implicazioni terapeutiche: mirare a un singolo bersaglio molecolare di solito fallisce perché il sistema tumorale è sufficientemente robusto da aggirare l'ostacolo. Le strategie più promettenti sono invece quelle di **co-targeting** di più hallmark capabilities (capacità caratteristiche del cancro) in combinazioni guidate dalla comprensione meccanicistica del sistema (multi-objective optimization methods).

Le caratteristiche che rendono così difficile il modeling computazionale del cancro includono:
- Eterogeneità intra-tumorale (cellule diverse all'interno dello stesso tumore)
- Plasticità fenotipica
- Interazione con il microambiente tumorale
- Evoluzione clonale sotto pressione selettiva terapeutica
- Rumore biologico intrinseco

---

### 1.5 Risposta robusta: restare o cambiare?

Dopo una perturbazione, la robustezza si può manifestare in due modi fondamentalmente diversi:

1. **Il sistema ritorna al proprio attrattore corrente** (*robust adaptation*): una volta rimossa la perturbazione, il sistema riprende il comportamento che aveva prima, come se nulla fosse accaduto.

2. **Il sistema si sposta verso un nuovo attrattore che mantiene le funzioni del sistema**: la struttura interna cambia in modo permanente, ma le funzioni essenziali vengono preservate nel nuovo stato.

Il concetto di **attrattore** (dal campo dei sistemi dinamici) è fondamentale qui. Un attrattore è uno stato verso cui il sistema evolve spontaneamente nel lungo periodo. Può essere:
- **Statico (point attractor):** il sistema raggiunge un equilibrio stabile e vi rimane.
- **Oscillatorio (periodic attractor):** il sistema oscilla perpetuamente tra diversi stati in modo ciclico.

Comprendere quali attrattori sono possibili, e le condizioni che permettono di passare da uno all'altro, è uno degli obiettivi principali dell'analisi computazionale dei sistemi complessi — e questo porta direttamente al cuore della teoria della biforcazione.

---

### 1.6 Quando la robustezza viene persa: bistabilità e punti critici

La robustezza non è una proprietà assoluta: esistono **punti critici** del sistema in cui una piccola variazione di un parametro di controllo provoca conseguenze qualitative sull'andamento asintotico del sistema. In questi punti critici — detti anche **punti di biforcazione** — il sistema perde la propria robustezza e transita verso un regime funzionale qualitativamente diverso.

I due fenomeni più importanti legati alla perdita di robustezza sono:

**1. La transizione da stato stazionario a regime oscillatorio (e viceversa)**
In molti sistemi biologici (come i pathway di segnalazione), il sistema può trovarsi in un regime di equilibrio stabile oppure esibirere oscillazioni persistenti. Queste due modalità corrispondono a due attrattori diversi. Ai punti critici, la piccola variazione di un parametro (es. la quantità di una proteina regolatrice) può spingere il sistema da un regime all'altro in modo netto e irreversibile, almeno localmente.

**2. Bistabilità**
Un sistema **bistabile** ha due stati stazionari stabili coesistenti per gli stessi valori dei parametri. A seconda delle condizioni iniziali (o di perturbazioni sufficientemente grandi), il sistema può trovarsi nell'uno o nell'altro attrattore. La bistabilità è biologicamente molto rilevante: spiega fenomeni come l'accensione (switch ON/OFF) di geni, la differenziazione cellulare irreversibile, o la scelta tra apoptosi e sopravvivenza.

**Connessione con i modelli computazionali:**
Comprendere dove si trovano questi punti critici — e quindi prevedere quando un sistema robusto può collassare — è uno dei principali obiettivi dell'analisi computazionale. Strumenti come la **Parameter Sweep Analysis**, la **Sensitivity Analysis** e la **Bifurcation Theory** (descritti nel §2) forniscono i metodi per identificare sistematicamente questi punti critici e per capire quali parametri ne sono i "fattori di controllo" critici.

---

## 2. Robustezza e Parametri del Modello

Quando si studia un sistema complesso tramite un modello matematico/computazionale, i parametri del modello giocano un ruolo cruciale. Essi possono rappresentare, in biologia, le quantità iniziali di molecole (es. il numero di molecole di una proteina) oppure le costanti cinetiche delle reazioni (quanto velocemente avviene una reazione chimica). La domanda fondamentale è: **quanto è sensibile il comportamento del sistema al variare di questi parametri?**

Esistono quattro approcci principali per rispondere a questa domanda, come sintetizzato in questo schema:

```
            PARAMETER          PARAMETER SWEEP
            ESTIMATION         ANALYSIS
            
            BIFURCATION        SENSITIVITY
            THEORY             ANALYSIS
```

---

### 2.1 Parameter Estimation

La **stima dei parametri** (*parameter estimation*) mira a trovare la parametrizzazione del modello che meglio si adatta ai dati sperimentali disponibili. È un problema di ottimizzazione: si cerca il minimo (globale) di una funzione di errore che misura la distanza tra le simulazioni del modello e le osservazioni reali.

Una domanda sottile ma fondamentale è: **ogni parametro del sistema è vincolato a un valore specifico e univoco, o esiste un intervallo di valori che garantiscono lo stesso comportamento del sistema?** In generale, i sistemi biologici mostrano una notevole tolleranza alla variazione parametrica — fenomeno noto come *sloppiness* del modello. Questo ha implicazioni importanti:

- I minimi locali nella funzione di errore possono non essere semplici artefatti numerici, ma riflettere fenomeni biologicamente significativi (come stati funzionali alternativi).
- Trovare il minimo globale non è sempre l'obiettivo finale: esplorare l'intero paesaggio dei parametri è spesso più informativo.

---

### 2.2 Parameter Sweep Analysis (PSA)

La **Parameter Sweep Analysis** (analisi di spazzolamento dei parametri) è un metodo computazionale che permette di analizzare sistematicamente l'effetto di diverse condizioni iniziali o valori parametrici sul comportamento del sistema.

**Come funziona:** si varia uno o più parametri all'interno di un intervallo fisso rispetto al valore di riferimento, e per ogni combinazione di valori si esegue una simulazione del modello. Poi si analizzano e classificano i risultati.

**PSA-1D (un parametro alla volta):**
- Campionamento lineare per le quantità molecolari
- Campionamento logaritmico per le costanti di reazione (perché coprono molti ordini di grandezza)

**PSA-2D (coppie di parametri):**
- Esplora simultaneamente come due parametri interagiscono nel determinare il comportamento del sistema
- Permette di costruire "mappe di fase" che mostrano quali combinazioni di valori portano a quali comportamenti (es. stato stazionario vs. oscillazioni)

**Campionamento con sequenze a bassa discrepanza (Low-Discrepancy Sequences):**
Il campionamento puramente casuale (*pseudo-random*, come quello generato dal Mersenne Twister) può essere inefficiente perché produce grappoli di punti in alcune zone e zone vuote in altre. Le **sequenze a bassa discrepanza** (come le **sequenze di Sobol**) producono punti molto meglio distribuiti nello spazio di ricerca, garantendo che ogni sottospazio riceva approssimativamente lo stesso numero di campioni. Il vantaggio è notevole soprattutto per campioni piccoli: già con 100 punti, le sequenze di Sobol coprono lo spazio in modo quasi uniforme, mentre i punti pseudo-casuali mostrano chiari grappoli e zone scoperte.

**Perché è utile?** La PSA permette di rispondere a domande come:
- In che intervallo di valori di un parametro il sistema mantiene il suo comportamento oscillatorio?
- A quale combinazione di due parametri corrisponde una transizione di fase qualitativa?
- Quali condizioni iniziali portano a quale attrattore?

**Esempio (pathway Ras/cAMP/PKA nel lievito):** la PSA-1D sulla quantità di Cdc25 (nell'intervallo 100-500 molecole, con valore di riferimento 300) ha mostrato che il regime oscillatorio stabile si osserva nell'intervallo 150 < Cdc25 < 400 molecole. Al di fuori di questo intervallo, le oscillazioni si smorzano e il sistema converge a uno stato stazionario. La PSA-2D su GTP e Cdc25 ha poi rivelato l'interazione tra questi due parametri, evidenziando quattro regimi qualitativamente diversi.

---

### 2.3 Sensitivity Analysis (SA)

La **Sensitivity Analysis** (analisi di sensibilità) si pone la domanda: **qual è l'effetto della variazione di un parametro di input su uno specifico output del modello?** Formalmente, si calcola il **coefficiente di sensibilità**:

$$S_{X_i, Y} = \frac{\partial Y}{\partial X_i}$$

dove $X_i$ è un parametro di input e $Y$ è l'output del modello considerato (es. frequenza delle oscillazioni, concentrazione all'equilibrio, ecc.). Questo coefficiente quantifica quanto cambia l'output al variare del parametro, permettendo di **classificare** i parametri dal più al meno influente.

**A cosa serve la SA?** Risponde a domande di cinque tipi:
1. **Validazione del modello:** Quali sono le conseguenze del cambiare un parametro di input? Il modello si comporta come ci aspettiamo biologicamente?
2. **Calibrazione del modello:** Quali parametri richiedono ulteriori ricerche per ridurre l'incertezza nell'output?
3. **Riduzione del modello:** Quali parametri non sono molto significativi e potrebbero essere rimossi senza alterare il comportamento?
4. **Robustezza del modello:** Quanto dipendono le previsioni del modello dai valori dei parametri?
5. **Analisi del controllo:** Quali parametri hanno la correlazione più alta con l'output del modello?

**SA locale vs. SA globale:**

| Tipo | Approccio | Quando usarla |
|------|-----------|---------------|
| **SA locale** | Esplora lo spazio dei parametri intorno a un valore nominale (neighborhood) | Modelli validati e stabili |
| **SA globale** | Esplora un ampio intervallo di valori (wide range) | Modelli in fase di bozza o instabili |

**One-Factor-At-a-Time (OAT):**
Il metodo più semplice di SA è quello che cambia un parametro alla volta, tenendo tutti gli altri al valore nominale, poi riportandolo al baseline e ripetendo per ogni parametro.

- *Vantaggi:* metodi consolidati, costo computazionale moderato.
- *Svantaggi:* non esplora le interazioni tra parametri; il risultato può dipendere dal valore baseline scelto; non copre l'intero spazio parametrico.

**Combination of Factors:**
Seleziona campioni di valori per tutti i parametri e genera il modello per ogni possibile combinazione.

- *Vantaggi:* copertura completa dello spazio, identifica interazioni tra parametri, indipendente dal baseline.
- *Svantaggi:* il numero di parametrizzazioni cresce esponenzialmente con il numero di parametri (curse of dimensionality); costi computazionali molto elevati.

**Esempio applicativo:** In una rete metabolica umana, la SA ha identificato che i flussi metabolici più influenti sul comportamento della rete sono quelli relativi ad amminoacidi essenziali (treonina, lisina) e alla vitamina biocitina.

---

### 2.4 Bifurcation Theory

La **teoria delle biforcazioni** è il ramo della matematica che studia come il comportamento qualitativo di un sistema dinamico cambia al variare di uno o più parametri di controllo.

**Definizione di biforcazione:** si ha una biforcazione quando una piccola variazione di uno o più valori parametrici provoca un cambiamento qualitativo nel comportamento del sistema — ad esempio, la transizione da oscillazioni a stato stazionario, o viceversa, oppure la comparsa o scomparsa di punti di equilibrio stabili.

Per comprendere la teoria delle biforcazioni è necessario avere familiarità con i **sistemi dinamici** e il concetto di **punto fisso** (o punto di equilibrio).

#### Sistemi lineari vs. non lineari

Prima di addentrarsi nei sistemi dinamici, è utile distinguere tra sistemi lineari e non lineari, perché i metodi analitici cambiano radicalmente.

**Sistemi lineari:** le equazioni del sistema sono combinazioni lineari delle variabili di stato. Sono trattabili analiticamente in modo completo: la soluzione generale si trova per sovrapposizione degli autovettori della matrice del sistema. Tuttavia, la linearità è un'approssimazione valida solo localmente attorno a un punto di equilibrio; la maggior parte dei sistemi reali (biologici, fisici, ingegneristici) è **non lineare**.

**Sistemi non lineari:** le equazioni contengono termini non lineari (prodotti di variabili, esponenziali, ecc.). Non ammettono in generale soluzioni analitiche esatte. Questo non significa però che il comportamento sia incomprensibile: la chiave è adottare un **approccio geometrico** anziché analitico.

Come afferma S.H. Strogatz:
> *"Le immagini sono spesso più utili delle formule per analizzare i sistemi non lineari. In molti casi l'informazione qualitativa è quella che ci interessa, e allora le immagini sono perfette."*

Lo **spazio delle fasi** (*phase space*) è la rappresentazione geometrica fondamentale: ogni stato del sistema è un punto nello spazio delle fasi, e le **traiettorie** mostrano come il sistema evolve nel tempo partendo da diverse condizioni iniziali. Ad esempio, la soluzione del pendolo non lineare — impossibile da esprimere in forma chiusa — è facilmente visualizzabile come una famiglia di curve nello spazio delle fasi (posizione × velocità), che rivela immediatamente regioni di oscillazione periodica, separatrici e punti di equilibrio.

Questo approccio geometrico permette di:
- Identificare punti fissi (equilibri) e la loro stabilità
- Comprendere la struttura globale degli attrattori
- Visualizzare come le traiettorie si organizzano nello spazio senza risolvere le ODE

#### Sistemi dinamici e punti fissi

Un sistema dinamico unidimensionale è descritto da un'equazione differenziale del tipo:

$$\dot{x} = f(x)$$

dove $\dot{x}$ rappresenta la velocità di cambiamento di $x$ nel tempo. L'interpretazione geometrica è quella di un **campo vettoriale sulla retta**: in ogni punto $x$, la funzione $f(x)$ ci dice con quale velocità e in quale direzione si muove il sistema.

Un **punto fisso** è un valore $x^*$ tale che $f(x^*) = 0$: in quel punto, il sistema è in equilibrio e non si muove.

- Un punto fisso è **stabile** se perturbazioni piccole decadono nel tempo e il sistema ritorna al punto fisso.
- Un punto fisso è **instabile** se perturbazioni piccole crescono nel tempo allontanando il sistema dal punto fisso.

**Esempio:** per $\dot{x} = x^2 - 1$, i punti fissi sono $x^* = +1$ (instabile) e $x^* = -1$ (stabile).

#### Le principali biforcazioni (sistemi 1D)

**Saddle-Node Bifurcation (biforcazione nodo-sella):**
È il meccanismo fondamentale con cui i punti fissi vengono creati e distrutti.

*Forma normale:* $\dot{x} = r + x^2$ (oppure $\dot{x} = r - x^2$)

Al variare del parametro $r$:
- Per $r < 0$: **due punti fissi** (uno stabile, uno instabile)
- Per $r = 0$: **un punto fisso semi-stabile** (punto di biforcazione)
- Per $r > 0$: **nessun punto fisso**

I due punti fissi si avvicinano, si scontrano al punto di biforcazione $r = 0$, e si annichilano mutuamente. Il **diagramma di biforcazione** visualizza questi punti fissi in funzione del parametro $r$, con le rami stabili indicati con linee continue e quelli instabili con linee tratteggiate.

**Transcritical Bifurcation (biforcazione transcritica):**
Descrive situazioni in cui un punto fisso deve esistere per tutti i valori del parametro e non può mai essere distrutto, ma può cambiare la propria stabilità.

*Forma normale:* $\dot{x} = rx - x^2$

- Per $r < 0$: $x^* = 0$ stabile, $x^* = r$ instabile
- Per $r = 0$: biforcazione — i due punti fissi si scambiano la stabilità
- Per $r > 0$: $x^* = 0$ instabile, $x^* = r$ stabile

È il meccanismo alla base di fenomeni dove una soluzione banale (es. assenza di malattia) perde stabilità e cede il passo a una soluzione non banale (es. endemia).

**Pitchfork Bifurcation (biforcazione a forcone):**
Appare in sistemi fisici caratterizzati da **simmetria** (es. sistemi con invarianza per riflessione $x \to -x$), per cui i punti fissi tendono ad apparire e scomparire in coppie simmetriche.

*Supercritical pitchfork:*
Forma normale: $\dot{x} = rx - x^3$

- Per $r < 0$: un punto fisso stabile ($x^* = 0$)
- Per $r = 0$: biforcazione
- Per $r > 0$: $x^* = 0$ diventa instabile; due nuovi punti fissi stabili $x^* = \pm\sqrt{r}$ appaiono simmetricamente

Questo è detto **effetto stabilizzante**: oltre la biforcazione, il sistema ha più opzioni stabili.

*Subcritical pitchfork:*
Forma normale: $\dot{x} = rx + x^3$

- Il comportamento è rovesciato: punti fissi instabili compaiono prima della biforcazione
- Detto **effetto destabilizzante**: può portare a salti discontinui nel comportamento del sistema (*hysteresis*)

**Hopf Bifurcation (biforcazione di Hopf):**
Nei sistemi a più dimensioni, la biforcazione di Hopf descrive la transizione da un punto fisso stabile a un ciclo limite stabile (oscillazione persistente). È particolarmente rilevante in biologia per spiegare l'insorgenza di oscillazioni in pathway di segnalazione.

#### Biforcazioni e storie d'amore: un esempio

Per rendere intuitiva la teoria delle biforcazioni, Strogatz propone l'esempio delle relazioni sentimentali modellate con equazioni differenziali accoppiate:

Supponiamo che $R(t)$ rappresenti l'amore/odio di Romeo per Giulietta e $J(t)$ quello di Giulietta per Romeo al tempo $t$. Valori positivi indicano amore, negativi odio.

**Caso 1 – Giulietta è "altalenante":**
Quanto più Romeo la ama, tanto più lei vuole allontanarsi; quando Romeo si scoraggia e si ritira, lei lo trova stranamente attraente. Romeo, invece, rispecchia i sentimenti di Giulietta.

Il modello è un sistema lineare del tipo:
$$\dot{R} = aJ, \quad \dot{J} = -bR \quad (a, b > 0)$$

La soluzione produce un ciclo oscillatorio eterno di amore e odio: almeno un quarto del tempo raggiungono amore simultaneo, ma non si stabilizza mai.

**Caso 2 – Entrambi si "riecheggiano":**
Se entrambi amplificano i sentimenti dell'altro ma tengono a cuore tre volte di più i propri sentimenti, il sistema può convergere verso un amore reciproco crescente o verso un odio reciproco crescente, a seconda delle condizioni iniziali. La separatrice (un autovettore del sistema) divide lo spazio delle fasi in regioni con esiti diversi.

**Caso 3 – Diversi gradi di cautela:**
La struttura degli attrattori cambia completamente al variare dei coefficienti del modello, mostrando come piccole variazioni nei "parametri della relazione" possano determinare esiti qualitativamente diversi.

Questi esempi illustrano un punto essenziale della teoria delle biforcazioni: **la struttura globale degli attrattori — e quindi il destino a lungo termine del sistema — dipende criticamente dai valori dei parametri**, spesso in modo non lineare e non intuitivo.

---

## 3. Caso Studio: Pathway Ras/cAMP/PKA nel Lievito

Un esempio concreto di come gli strumenti computazionali descritti vengono applicati a un sistema biologico reale è il pathway **Ras/cAMP/PKA** nel lievito (*Saccharomyces cerevisiae*).

### Contesto biologico

Il lievito risponde alla presenza di glucosio attraverso 5 pathway interlacciati che producono massicce modificazioni trascrizionali (cambiamenti nell'espressione genica). Al centro di questa risposta c'è il pathway Ras/cAMP/PKA, che controlla:
- Regolazione del metabolismo
- Resistenza allo stress
- Progressione del ciclo cellulare

### Struttura molecolare del pathway

Il pathway funziona secondo questa logica (semplificata):

1. Le **proteine Ras** (GTPasi) si trovano in due stati: attivo (legato a GTP) e inattivo (legato a GDP).
2. **Cdc25** (fattore di scambio GDP→GTP) attiva Ras; **Ira2** (proteina attivatrice di GTPasi) inattiva Ras.
3. **Ras2-GTP** attiva l'adenilato ciclasi **Cyr1**, che catalizza la sintesi di cAMP.
4. Il **cAMP** attiva la proteina chinasi **PKA** (Protein Kinase A).
5. Il cAMP viene degradato dalle due fosfodiesterasi **Pde1** e **Pde2**.
6. **PKA** esercita retroazioni (feedback):
   - Attivazione positiva di Pde1 (→ degrada cAMP → loop negativo)
   - Attivazione positiva di Ira2 (→ inattiva Ras → loop negativo)
   - Inibizione negativa di Cdc25 (→ riduce attivazione di Ras → loop negativo)

### Il modello computazionale

Il modello matematico del pathway comprende **33 specie molecolari** e **39 reazioni**. La domanda scientifica che ha guidato la costruzione del modello era: *In quali condizioni si instaurano oscillazioni nel pathway? Qual è il ruolo di Cdc25, Ira2 e dei nucleotidi GTP/GDP?*

Le evidenze sperimentali indirette di oscillazioni erano state osservate nel nucleo-citoplasma shuttling di Msn2 (un target a valle di PKA).

### Risultati della PSA e della SA

**Ruolo del feedback:**
Le simulazioni mostrano che l'attivazione di **entrambi** i controlli di feedback (su Cdc25 E su Ira2) è necessaria per ottenere regimi oscillatori stabili del cAMP. Rimuovere uno dei due feedback elimina le oscillazioni.

**Ruolo di Cdc25 (PSA-1D):**
- Oscillazioni stabili si osservano nell'intervallo 150 < Cdc25 < 400 molecole
- Al di fuori di questo intervallo: oscillazioni smorzate → stato stazionario
- La simulazione stocastica rivela fluttuazioni biologiche intorno allo stato stazionario anche fuori dall'intervallo oscillatorio

**Differenza tra simulazione stocastica e deterministica:**
Questo pathway illustra in modo eccellente la differenza tra i due approcci simulativi. La simulazione deterministica (equazioni differenziali ordinarie) rivela gli attrattori del sistema medio, ma non cattura le fluttuazioni. La simulazione stocastica (algoritmi come il Gillespie SSA) tiene conto del fatto che le molecole sono entità discrete e le reazioni eventi probabilistici — fondamentale quando le quantità molecolari sono piccole (come nei pathway di segnalazione intracellulare).

**Ruolo dei nucleotidi guaninati (GTP/GDP) — PSA-1D:**
Prima di analizzare la combinazione dei due fattori, è stata condotta una PSA-1D sul GTP separatamente. La quantità iniziale di GTP influenza le oscillazioni in modo differente a seconda della quantità di Cdc25:
- Con Cdc25 al valore standard (300 molecole): variare GTP modifica l'ampiezza delle oscillazioni ma non sempre elimina il regime oscillatorio.
- Con Cdc25 in over-espressione (500 molecole): la variazione di GTP produce effetti qualitativamente diversi, spostando prima il sistema fuori dal regime oscillatorio stabile.
Questa analisi monodimensionale ha motivato l'esplorazione bidimensionale combinata.

**Ruolo dei nucleotidi guaninati (GTP/GDP) — PSA-2D:**
La PSA-2D su GTP e Cdc25 ha rivelato quattro regimi qualitativamente distinti, mappati in uno spazio bidimensionale. Questo tipo di analisi è possibile solo perché si possono eseguire decine di migliaia di simulazioni grazie al **GPU computing**: 65.000 simulazioni in 2 ore su GPU, contro sole 200 simulazioni in 2 ore su CPU.

| Regime | GTP | Cdc25 | Comportamento |
|--------|-----|-------|---------------|
| I | basso | basso | nessuna oscillazione, stato stazionario |
| II | alto | basso | oscillazioni smorzate |
| III | basso | alto | oscillazioni smorzate |
| IV | alto | alto | oscillazioni stabili |

**Ruolo della fosfodiesterasi Pde1 (PSA-1D):**
Un'ulteriore PSA-1D è stata condotta sulla costante di fosforilazione di **Pde1** (parametro $c_{26}$, nell'intervallo $[1.0 \times 10^{-9}, 1.0 \times 10^{-3}]$, con valore di riferimento $1.0 \times 10^{-6}$). Pde1 è uno degli enzimi che degrada il cAMP ed è attivato da PKA (loop di feedback negativo).

I risultati mostrano che:
- Per valori di $c_{26}$ molto bassi (fosforilazione lenta di Pde1): il feedback negativo tramite Pde1 è debole, e il cAMP si accumula — il sistema tende verso uno stato stazionario ad alta concentrazione di cAMP.
- Per valori di $c_{26}$ vicini al riferimento: il feedback è bilanciato e si instaurano oscillazioni stabili.
- Per valori di $c_{26}$ molto alti (fosforilazione rapida di Pde1): il feedback negativo è troppo forte, il cAMP viene abbattuto rapidamente e il sistema converge a uno stato stazionario a bassa concentrazione.

Questa analisi sottolinea come il bilanciamento preciso dei loop di feedback (Pde1, Cdc25, Ira2) sia necessario per mantenere il regime oscillatorio — e come la robustezza di questo regime dipenda da un intervallo relativamente ristretto di valori parametrici.

**Significato biologico delle oscillazioni:**
Le oscillazioni in questo pathway potrebbero estendere il range regolatorio del sistema attraverso la **modulazione di frequenza** (frequency-modulated signaling): la PKA — che controlla il 90% dei geni regolati dal glucosio nel lievito — può codificare informazioni diverse in base alla frequenza delle oscillazioni di cAMP, non solo alla loro ampiezza.

---

## 4. AI e Deep Learning per la Diagnosi Cardiovascolare (CMR)

### 4.1 Background medico: MRI e CMR

#### Risonanza Magnetica (MRI)

La **Risonanza Magnetica per Immagini** (MRI, Magnetic Resonance Imaging) è una tecnica diagnostica non invasiva che sfrutta le proprietà magnetiche dei **protoni di idrogeno** abbondanti nei tessuti biologici. Il meccanismo di funzionamento si articola in tre fasi:

1. **Allineamento:** I protoni, che si comportano come piccoli magneti rotanti (*spin*), vengono allineati da un forte campo magnetico esterno (1.5 Tesla o 3 Tesla nei dispositivi clinici).
2. **Eccitazione:** Un impulso di radiofrequenza (RF) capovolge l'orientamento dei protoni rispetto al campo.
3. **Rilassamento:** Quando l'impulso RF viene rimosso, i protoni ritornano all'equilibrio emettendo un segnale elettromagnetico. La *velocità* con cui ritornano all'equilibrio dipende dall'ambiente molecolare specifico di ciascun tessuto, permettendo di differenziare i tessuti nell'immagine.

#### Risonanza Magnetica Cardiovascolare (CMR)

La **CMR** (Cardiovascular Magnetic Resonance) è considerata il gold standard per la valutazione non invasiva dell'anatomia cardiaca e della caratterizzazione del miocardio (il muscolo cardiaco).

Le tecniche convenzionali includono:
- **Late Gadolinium Enhancement (LGE):** utilizza agenti di contrasto (gadolinio) che accorciano il T1 per visualizzare fibrosi e necrosi. Distingue lesioni ischemiche (endocardiali) da lesioni non ischemiche (mid-wall/epicardiali).
- **T2-weighted Imaging:** standard per rilevare edema miocardico e infiammazione.

**Limitazioni delle tecniche qualitative convenzionali:**
- Natura qualitativa: l'interpretazione è soggettiva e dipende dall'esperienza del radiologo.
- Sensibilità limitata per patologie diffuse (manca un tessuto di riferimento sano all'interno dell'immagine).

---

### 4.2 Il paradigma quantitativo: T1 e T2 mapping

Per superare i limiti delle tecniche qualitative, si è sviluppato il **paradigma quantitativo** basato sul *parametric mapping*: anziché produrre immagini di contrasto soggettive, si producono mappe pixel-per-pixel dei valori assoluti (in millisecondi) dei tempi di rilassamento.

#### T1 Mapping (rilassamento longitudinale, spin-lattice)

Il **T1** è il tempo di rilassamento longitudinale: misura la velocità con cui la magnetizzazione recupera lungo l'asse longitudinale (asse z) dopo l'eccitazione. Fisicamente, riflette il trasferimento di energia dai protoni eccitati alla struttura molecolare circostante (*lattice*).

**Metodologia:** si acquisiscono immagini multiple a diversi intervalli dopo impulsi di inversione o saturazione, e si ricostruisce pixel per pixel la mappa dei valori T1.

**Significato clinico:**
- T1 nativo elevato → edema, amiloidosi, fibrosi diffusa
- T1 nativo ridotto → infiltrazione lipidica (Anderson-Fabry), depositi di ferro
- **ECV (Extracellular Volume):** calcolato combinando T1 nativo e post-contrasto, quantifica l'espansione della matrice extracellulare (indicatore di fibrosi)

#### T2 Mapping (rilassamento trasversale, spin-spin)

Il **T2** è il tempo di rilassamento trasversale: misura il decadimento della magnetizzazione nel piano trasversale dovuto alla perdita di coerenza di fase tra spin vicini.

**Metodologia:** si fittano le curve di decadimento del segnale acquisite a diversi tempi di eco (TE).

**Vantaggi diagnostici:**
- Quantificazione obiettiva del contenuto idrico miocardico
- Elimina artefatti da flusso lento
- Chiave per miocarditi e rigetto acuto post-trapianto

#### Sfide tecniche e standardizzazione

Le misurazioni di T1 e T2 sono influenzate da:
- Intensità del campo magnetico (1.5 T vs 3 T)
- Sequenze specifiche del costruttore (vendor-specific)
- Parametri di acquisizione

I valori di riferimento variano anche con età, sesso e stato fisiologico del paziente. Questo rende necessaria la **calibrazione centro-specifica** — ogni ospedale deve stabilire i propri intervalli di normalità — complicando il confronto tra centri diversi e il monitoraggio longitudinale.

---

### 4.3 Barriere all'adozione clinica dell'AI per CMR

Per trasferire con successo l'AI nell'ambiente clinico reale occorre affrontare quattro barriere principali:

1. **Reference Variability:** i valori di riferimento dipendono fortemente dall'intensità del campo magnetico, dalla sequenza di acquisizione e dai parametri scelti.

2. **Data Scarcity and Imbalance:** la disponibilità di dataset annotati di alta qualità è limitata (l'annotazione richiede expertise medico costoso); la prevalenza naturale di certe condizioni porta a **class imbalance** (molti più soggetti sani che malati).

3. **The "Black Box" Problem:** i modelli deep learning complessi non spiegano come arrivano alle loro decisioni. Questo mina la fiducia clinica.

4. **Overconfidence Risk:** i modelli possono produrre previsioni con alta confidenza anche quando sono errati, senza fornire una misura affidabile dell'incertezza — pericoloso in ambito clinico dove si prendono decisioni ad alto rischio.

---

### 4.4 Apprendimento supervisionato

#### Panoramica dei paradigmi di apprendimento

Prima di entrare nel dettaglio del lavoro, è utile inquadrare i tre paradigmi di apprendimento automatico rilevanti in questo contesto:

| Paradigma | Dati usati | Caratteristiche | Limitazioni |
|-----------|-----------|-----------------|-------------|
| **Supervised** | Solo dati etichettati $(x, y)$ | Alta accuratezza; logica diretta | Costoso da annotare in ambito medico |
| **Unsupervised** | Solo dati non etichettati $(x)$ | Nessun bisogno di annotazioni | Non controlla direttamente il task diagnostico |
| **Semi-Supervised (SSL)** | Piccolo set etichettato + grande pool non etichettato | Sfrutta dati abbondanti non annotati | Richiede assunzioni sulla struttura dei dati |

In ambito medico, le annotazioni di esperti sono **costose e scarse** (richiedono expertise radiologico specializzato), mentre i dati non annotati sono abbondanti. Questo rende il SSL particolarmente attraente — e motiva la progressione nel lavoro da approccio supervisionato → semi-supervisionato → ensemble.

#### Dataset e struttura

Il dataset proviene dall'Ospedale Universitario – Istituto di Radiologia dell'Università di Padova. Ogni paziente contribuisce con **6 immagini**:
- 3 mappe T1 (longitudinali): slice basale, mid-cavity, apicale (secondo le linee guida AHA)
- 3 mappe T2 (trasversali): stesse tre slice

**Assegnazione delle etichette cliniche:**
Le etichette a livello paziente vengono aggregate dalle evidenze a livello di immagini per rispecchiare gli standard diagnostici clinici (sensibilità clinica): un paziente viene classificato come "malato" se almeno una sua immagine mostra evidenza di patologia.

**Suddivisione dei dati:**
- Training: 60%
- Validation: 20%
- Testing: 20%

**Mitigazione del data leakage:** tutte le immagini appartenenti allo stesso paziente sono confinate nello stesso split. Questo è fondamentale: se immagini dello stesso paziente finissero sia nel training che nel test set, il modello imparerebbe a "riconoscere" quel paziente specifico anziché generalizzare a nuovi pazienti.

#### Architetture neurali

**CNN (Convolutional Neural Networks):**
Le CNN sfruttano la connettività locale e il bias induttivo spaziale: i filtri convoluzionali operano su porzioni locali dell'immagine e vengono condivisi su tutta l'immagine (weight sharing). Questo le rende eccellenti per l'estrazione gerarchica di feature (bordi → forme → strutture complesse). Modelli utilizzati: EfficientNet (B4, V2-RW-S/M), ResNet, ConvNeXt.

**ViT (Vision Transformers):**
I Vision Transformers adattano l'architettura Transformer (originariamente sviluppata per il NLP) alle immagini. L'immagine viene divisa in patch (blocchi di pixel), ognuno dei quali viene trattato come un "token". Il meccanismo di **self-attention** permette a ogni patch di "guardare" tutte le altre contemporaneamente, catturando dipendenze a lungo raggio nell'immagine che le CNN faticano a modellare con i loro filtri locali. Modello utilizzato: ViT-Base.

**Obiettivo del confronto CNN vs ViT:**
Confrontare la "efficienza locale" delle CNN con la "capacità rappresentativa globale" dei Transformer, per capire quali strutture portano a diagnosi più accurate e calibrate su immagini di T1/T2 mapping.

#### Data Augmentation

Per contrastare l'overfitting e migliorare la robustezza del modello a variazioni di acquisizione e anatomia, viene applicata una pipeline sistematica di **data augmentation**. L'obiettivo è simulare realisticamente la variabilità clinica (differenze di scanner, postura del paziente, artefatti) in modo che il modello apprenda feature distribuita e contestuale anziché affidarsi a cue localizzati.

La pipeline include trasformazioni geometriche (rotazioni, flip orizzontali/verticali, zoom casuali) e trasformazioni di intensità (variazioni di contrasto e luminosità) per simulare differenze tra centri e scanner diversi.

**Enhancements specifici per i Vision Transformer:**
- **Gaussian blur** (sfocatura gaussiana): simula la variabilità di risoluzione tra sequenze di acquisizione diverse, forzando la self-attention a catturare strutture a scale multiple.
- **Aggressive RandomResizedCrop:** ritaglio casuale a scala variabile che sfrutta la sensibilità spaziale della self-attention, evitando che il modello si ancori a posizioni fisse nell'immagine.

L'effetto complessivo è che il modello è costretto a imparare **rappresentazioni distribuite e context-aware**, resistenti agli artefatti di acquisizione tipici delle immagini CMR in contesti clinici reali.

#### Risultati sperimentali: approccio supervisionato

**Risultati a livello di immagine (image-level):**
Le architetture moderne (EfficientNetV2-m, ViT-Base) superano costantemente le architetture legacy (ResNet-50). Tuttavia, i valori di ROC-AUC rimangono moderati su tutti i singoli modelli. La causa principale è la **strategia di labeling clinicamente motivata**: le etichette vengono propagate dal paziente a tutte le sue slice (una slice è "malata" se il paziente è malato, indipendentemente da quella slice specifica). Questo introduce noise a livello di immagine, rendendo difficile la discriminazione a quel livello.

**Risultati a livello di paziente (patient-level) e il Paradosso ConvNeXt:**
Aggregando le predizioni delle singole slice per produrre una diagnosi a livello paziente, le performance migliorano. Emerge però un **paradosso significativo** con ConvNeXt-Base:

| Metrica | ConvNeXt-Base | Interpretazione |
|---------|--------------|-----------------|
| Accuracy | Più alta | Ottima capacità di rilevamento |
| F1-Score | Più alto | Buona precision-recall |
| ROC-AUC | Più bassa | Scarsa calibrazione delle probabilità |

Questo "paradosso" dimostra che **alta accuratezza può coesistere con scarsa calibrazione**: il modello è molto capace di classificare correttamente, ma le sue probabilità predette non sono ben allineate con le frequenze empiriche — è overconfident. Questa è una problematica critica in ambito clinico, dove la stratificazione del rischio richiede probabilità affidabili, non solo classificazioni binarie.

La stessa tendenza si osserva per la famiglia EfficientNet. Questa constatazione motiva direttamente l'adozione di tecniche come il **Label Smoothing**, la **Focal Loss** e il **Temperature Scaling** per migliorare la calibrazione, descritti nella sezione successiva.

---

### 4.5 Loss function: Focal Loss e Label Smoothing

Per addestrare modelli robusti su dataset sbilanciati e con etichette propagate (fonte di rumore), vengono combinate due tecniche speciali.

#### Focal Loss

La **Cross-Entropy standard** tende a concentrarsi sui campioni "facili" (ad esempio, i numerosi soggetti sani che il modello classifica correttamente con alta confidenza fin dai primi passi del training), ignorando i campioni difficili e rari — esattamente i casi patologici che ci interessano di più clinicamente.

La **Focal Loss** introduce un fattore modulante $(1 - p_t)^\gamma$ che riduce automaticamente il contributo dei campioni "facili" alla loss:

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

Dove:
- $\gamma$ (focusing parameter): controlla quanto enfatizzare i campioni difficili e mal classificati. Per $\gamma=0$ si riduce alla cross-entropy standard.
- $\alpha_t$ (class weight): bilancia il contributo delle classi sottorappresentate.

**Razionale clinico:** down-pesa i casi ovvi (sani che il modello riconosce facilmente) e dirige le risorse di apprendimento verso i casi ambigui e le patologie rare.

#### Label Smoothing

Le reti neurali profonde tendono a predire probabilità "dure" (0 o 1), portando a **overfitting** e **scarsa calibrazione**: il modello è troppo sicuro di sé anche quando sbaglia.

**La calibrazione** misura quanto bene le probabilità predette rispecchiano le frequenze empiriche: un modello calibrato che predice 0.7 di probabilità di malattia dovrebbe essere corretto circa il 70% delle volte che lo fa.

Il **Label Smoothing** ridistribuisce una piccola quota $\varepsilon$ di probabilità dalla classe vera a tutte le altre:

$$\tilde{y}_k = (1-\varepsilon) \cdot y_k + \varepsilon / K$$

dove $K$ è il numero di classi. Questo impedisce al modello di formare decisioni eccessivamente nette, migliorando la regolarizzazione e la calibrazione.

#### Il framework ibrido (Hybrid Loss)

I due approcci vengono combinati in una combinazione convessa:

$$\mathcal{L} = \beta \cdot \mathcal{L}_{FL} + (1-\beta) \cdot \mathcal{L}_{LS}$$

Il parametro $\beta$ viene ottimizzato tramite **Optuna** (framework Bayesiano per l'ottimizzazione degli iperparametri, descritto in §4.10) per trovare il miglior bilanciamento tra "focus sui campioni difficili" e "calibrazione/regolarizzazione".

I pesi di classe $\alpha_t$ sono calcolati come inverso delle frequenze di classe nel training set, assicurando che le classi minoritarie contribuiscano proporzionalmente di più alla loss totale.

---

### 4.6 Apprendimento semi-supervisionato (SSL)

#### Fondamenti teorici del SSL

Il **Semi-Supervised Learning** (SSL) combina un piccolo dataset etichettato con un grande pool di dati non etichettati. È particolarmente prezioso in ambito medico, dove le annotazioni di esperti sono costose e scarse, mentre i dati non annotati sono abbondanti.

Il SSL si basa su tre assunzioni teoriche:
1. **Smoothness assumption:** punti vicini nello spazio delle feature condividono probabilmente la stessa etichetta.
2. **Cluster assumption:** i confini decisionali dovrebbero passare nelle regioni a bassa densità tra i cluster, non attraversarli.
3. **Manifold assumption:** le immagini ad alta dimensionalità giacciono su una varietà (manifold) a bassa dimensionalità, dove le variazioni clinicamente rilevanti cambiano in modo continuo.

#### Self-Training con Soft Pseudo-Labels

La strategia più semplice di SSL è il **self-training**: il modello, addestrato sui dati etichettati, genera *pseudo-etichette* per i dati non etichettati, che vengono poi usati nel training successivo.

**Problema delle hard labels:** il pseudo-labeling tradizionale usa il $\arg\max$ dell'output softmax, collassando la distribuzione di probabilità in un vettore one-hot ($[0,1]$ o $[1,0]$). Questo ignora completamente la confidenza del modello: una predizione al 51% viene trattata identicamente a una al 99%, amplificando l'errore.

**Soluzione: Soft Pseudo-Labels**
Si usa l'intera distribuzione di output $\hat{p}_j$ come target. Un soft target come $[0.6, 0.4]$ rappresenta l'incertezza epistemica del modello e impedisce al modello di sovra-impegnarsi su una classe potenzialmente errata. Dal punto di vista teorico, questo si allinea con il **Bayesian Learning**, dove le etichette sono trattate come variabili latenti nel simplesso di probabilità $\Delta^K$.

**EMA (Exponential Moving Average):**
Le soft labels per ogni campione $j$ non vengono sostituite ad ogni iterazione, ma evolvono tramite una media mobile esponenziale:

$$s_j^{(t+1)} = \alpha \cdot s_j^{(t)} + (1-\alpha) \cdot \hat{p}_j^{(t)}$$

Questo filtra il rumore stocastico dei mini-batch e crea un effetto di *temporal ensembling*: la label riflette le previsioni medie di versioni multiple del modello nel tempo, teoricamente più accurate di qualsiasi singola previsione.

**Punto importante:** l'EMA è applicata **sia ai dati non etichettati che a quelli etichettati**. Per i dati non etichettati, l'EMA affina gradualmente la pseudo-label. Per i dati etichettati, la comprensione evolutiva del modello "ammorbidisce" le etichette ground-truth: anziché imparare da etichette fisse one-hot (0 o 1), il modello integra la sua comprensione crescente del campione, ottenendo una forma di *regolarizzazione appresa* che riduce l'overfitting alle etichette rumorose.

**Limitazioni del self-training di base:**
- **Error propagation:** nella struttura gerarchica delle etichette, una singola slice mal classificata può "contaminare" il segnale di training nei round successivi.
- **Confirmation bias:** senza validazione esterna, il modello tende a rinforzare le proprie previsioni errate o ambigue. La soluzione richiede un **Expert-in-the-loop** (Active Learning).

#### Mean Teacher

Il paradigma **Mean Teacher** supera i limiti del self-training di base con un'architettura a doppio modello:

- **Student** ($\phi$): il modello attivo, ottimizzato tramite SGD.
- **Teacher** ($\theta$): un "ensemble temporale" dello Student, i cui pesi sono l'EMA dei pesi dello Student:
$$\theta^{(t+1)} = \lambda \cdot \theta^{(t)} + (1-\lambda) \cdot \phi^{(t)}$$

Il Teacher è dunque una versione "smussata" dello Student: media i pesi del modello attraverso il tempo, producendo previsioni più stabili e meno rumorose. Questo crea un target di consenso che guida lo Student verso confini decisionali più lisci.

**Obiettivo di training composito:**
$$\mathcal{L} = \mathcal{L}_{sup} + \lambda_{cons} \cdot \mathcal{L}_{cons}$$

- $\mathcal{L}_{sup}$: soft cross-entropy tra i logit dello Student e le etichette ground-truth
- $\mathcal{L}_{cons}$: KL divergence tra le distribuzioni dello Student e del Teacher (incentiva la consistenza)

**Confidence Masking:** la consistency loss viene applicata solo se la confidenza del Teacher supera una soglia $\tau \in [0.7, 0.9]$. Questo evita di propagare l'incertezza del Teacher su campioni in cui esso stesso non è sicuro.

**Post-Training Calibration:** dopo il training, si applica il **Temperature Scaling** per assicurare che le probabilità predette rispecchino il rischio diagnostico reale, evitando l'overconfidence.

**Risultati:** il Mean Teacher supera consistentemente il self-training di base, raggiungendo alta affidabilità diagnostica con Balanced Accuracy > 0.84 nella maggior parte dei trial.

---

### 4.7 Model Ensembling

#### Razionale

L'**ensemble** combina le previsioni di più modelli per ridurre la varianza e migliorare la robustezza. Dal punto di vista della teoria statistica, l'ensembling riduce principalmente la varianza (decomposizione bias-varianza) mediando gli errori non correlati dei modelli diversi.

**Interpretazione Bayesiana:** l'ensemble è un'approssimazione Monte Carlo della distribuzione predittiva posteriore — in altre parole, campionare da diversi modelli è un modo di approssimare l'incertezza epistemica del modello ideale.

#### Eterogeneità architetturale

L'ensemble combina:
- CNN di diverse famiglie (EfficientNet, ConvNeXt, ResNet)
- Vision Transformer (ViT)
- Modello semi-supervisionato (Mean Teacher)

Questa diversità architetturale assicura che gli errori dei modelli siano il più possibile non correlati — condizione necessaria per il successo dell'ensemble.

#### Calibrazione e Temperature Scaling

Prima di aggregare le previsioni, è essenziale che tutti i modelli siano sulla stessa "scala probabilistica". I modelli deep learning tendono a essere overconfident: il loro output softmax non rappresenta probabilità calibrate (il modello dice 0.95 di confidenza ma è corretto solo il 70% delle volte).

Il **Temperature Scaling** è un metodo post-hoc che risolve questo problema dividendo i logit del modello per un parametro di temperatura $T$ (appreso su un validation set):

$$\hat{p}_k = \frac{e^{z_k / T}}{\sum_j e^{z_j / T}}$$

Per $T > 1$ le probabilità vengono "smussate" (riduce l'overconfidence), per $T < 1$ vengono affilate.

#### Strategie di aggregazione

1. **Simple Averaging:** $\bar{p}(x) = \frac{1}{M} \sum_{m=1}^{M} p_m(x)$ — tratta tutti i modelli ugualmente.
2. **Weighted Averaging:** $\bar{p}(x) = \sum_m w_m \cdot p_m(x)$ con $\sum_m w_m = 1$ — assegna più peso ai modelli con ROC-AUC più alta sul validation set.

I risultati mostrano che la diversità architetturale è più influente della specifica strategia di aggregazione: il weighted averaging non supera significativamente il simple averaging quando i modelli sono sufficientemente diversi.

**Risultati a livello di immagine (image-level):**
L'ensemble mitiga i bias individuali di ciascun backbone. La complementarità architetturale tra CNN (efficienza locale) e ViT (contesto globale) porta a una baseline di predizione più stabile di qualsiasi modello singolo. Notabilmente, l'averaging uniforme è quasi equivalente al weighted averaging: quando i modelli sono sufficientemente diversi, la diversità architetturale conta più della ponderazione precisa.

**Risultato clinico a livello paziente (patient-level):**
L'aggregazione delle predizioni a livello di slice porta a un **salto massiccio nell'affidabilità diagnostica a livello paziente**. Il sistema dimostra alta efficacia per la stratificazione del rischio. Questo conferma che **la diversità architetturale è più influente della specifica strategia di aggregazione**: un ensemble eterogeneo (CNN + ViT + SSL) con averaging uniforme e calibrazione è sufficiente per ottenere prestazioni clinicamente rilevanti.

---

### 4.8 Explainability (XAI)

#### Il contesto normativo: EU AI Act

L'**AI Act** dell'Unione Europea è il primo framework legale comprensivo per l'AI. I sistemi AI in sanità sono classificati come **High-Risk** (Art. 6) perché influenzano la vita e la salute delle persone.

Articoli rilevanti:
- **Art. 13 (Transparency):** i sistemi devono essere progettati in modo che gli utenti possano interpretare l'output e usarlo appropriatamente. I modelli opachi sono una responsabilità legale.
- **Art. 86 (Right to Explanation):** i pazienti hanno il diritto a una "spiegazione chiara e significativa" di come il sistema AI ha raggiunto una decisione che li riguarda.

**Implicazione pratica:** la XAI non è più un "nice-to-have" ma un **requisito obbligatorio** per i sistemi AI ad alto rischio in Europa.

#### Explainability vs. Interpretability

Questi due termini vengono spesso usati come sinonimi, ma hanno significati distinti:

- **Explainability (post-hoc):** si "tortura il modello" per estrarne informazioni dopo che il processo è avvenuto. Il modello è ancora una scatola nera, ma si generano spiegazioni per giustificare le sue decisioni *a posteriori* (es. Grad-CAM).

- **Interpretability (ex-ante):** il modello è progettato per essere intrinsecamente intelligibile — la sua logica è trasparente prima ancora che faccia una previsione ("white-box by design").

**Trade-off:** la explainability prioritizza l'accuratezza grezza in domini complessi; la interpretability può sacrificare un po' di accuratezza per la chiarezza totale, spesso più preziosa in contesti clinici ad alto rischio.

#### Grad-CAM

Il **Grad-CAM** (Gradient-weighted Class Activation Mapping) è una tecnica post-hoc di visual attribution che genera **heatmap** che evidenziano le regioni dell'immagine più influenti per la decisione del modello.

**Come funziona:** si calcolano i gradienti della classe predetta rispetto all'attivazione dell'ultimo layer convoluzionale, si pesano i canali di attivazione per queste grandezze e si proietta la mappa risultante sull'immagine di input.

**Applicazione clinica:** le heatmap preservano l'anatomia del cuore, permettendo ai medici di verificare se l'attivazione del modello si concentra sul miocardio (come dovrebbe) e di capire *perché* il modello ha classificato un'immagine in un certo modo.

#### La crisi della explainability: una realtà scomoda

Una collaborazione con i radiologi dell'Ospedale di Padova ha rivelato una serie di problemi fondamentali con i metodi XAI attuali (presentato alla 3rd World Conference of Explainable AI, 2025):

- **Inconsistency:** ogni metodo XAI produceva spiegazioni diverse per la stessa immagine.
- **No "Gold Standard":** nessun metodo si è dimostrato chiaramente superiore agli altri.
- **Experience Bias:** il metodo preferito correlava con l'esperienza del medico (i veterani erano più conservatori/scettici).

**Conclusione fondamentale:** alta accuratezza non garantisce spiegazioni affidabili. La XAI spesso non riesce a colmare il gap tra la logica della "scatola nera" e l'intuizione clinica. Questo motiva la ricerca verso modelli intrinsecamente interpretabili.

#### pyFUME: verso l'interpretabilità by design

**pyFUME** (Python Fuzzy MEasure) è un framework open-source per la costruzione di **sistemi di inferenza fuzzy** (Fuzzy Inference Systems, FIS) che siano intrinsecamente interpretabili. A differenza dei modelli deep learning che producono spiegazioni post-hoc (come Grad-CAM), i modelli basati su pyFUME sono *white-box by design*: la loro logica è trasparente prima ancora che venga fatta una predizione.

**Come funziona:**
Un sistema fuzzy rappresenta la conoscenza tramite **regole IF-THEN** in linguaggio naturale, del tipo:
- *IF T1 è "elevato" AND età è "giovane" THEN diagnosi è "probabile patologia"*

Le variabili linguistiche (es. "elevato", "giovane") sono definite da **funzioni di appartenenza** (*membership functions*) che consentono una transizione graduale tra categorie (a differenza della logica classica binaria). Questo rispecchia meglio la natura continua delle misurazioni mediche.

**Perché è rilevante per la fairness:**
La struttura esplicita delle regole fuzzy permette di:
1. **Verificare** che il modello non stia usando attributi protetti (genere, età, etnia) in modo discriminatorio.
2. **Auditare** le decisioni in modo comprensibile da medici e pazienti.
3. **Garantire equità by design**: le regole possono essere vincolate esplicitamente per assicurare prestazioni bilanciate su diversi sottogruppi.

**Trade-off accuratezza vs. interpretabilità:**
I sistemi fuzzy possono avere accuratezza leggermente inferiore ai modelli deep learning su dataset complessi. Tuttavia, in contesti clinici ad alto rischio, questa perdita marginale di accuratezza è spesso accettabile in cambio di trasparenza totale, conformità normativa (EU AI Act Art. 13 e 86) e fiducia da parte di medici e pazienti.

---

### 4.9 Fairness nell'AI clinica

#### Il problema dell'"accuracy trap"

Un modello può essere accurato *in media* ma **sistematicamente biasato** verso specifiche sottopopolazioni (per genere, età, etnia). Un modello che raggiunge il 90% di accuratezza globale potrebbe avere il 70% di accuratezza sulle donne e il 95% sugli uomini — un risultato inaccettabile in ambito clinico.

#### Fairness by Design

L'approccio proposto sposta il paradigma da "Deep Learning for Performance" a "Interpretable AI for Fairness-by-Design":

- **Oltre l'accuratezza:** l'accuratezza è priva di significato se ottenuta a spese di certi gruppi di pazienti.
- **Trust & Stakeholders:** l'affidabilità in contesti clinici ad alto rischio richiede esiti equi.
- **Integrazione:** combinare Deep Learning con framework basati sulla logica (come pyFUME) per creare sistemi che siano sia precisi che equi e interpretabili.

L'AI non deve essere solo uno strumento di precisione, ma un veicolo per un processo decisionale responsabile e rispettoso delle differenze individuali.

---

### 4.10 Dettagli tecnici: ottimizzazione degli iperparametri e fine-tuning

#### Hyperparameter Optimization (HPO) con Optuna

Le prestazioni del Deep Learning dipendono fortemente dagli **iperparametri** (learning rate, batch size, parametri di regolarizzazione, ecc.), che interagiscono in modo non lineare. La ricerca manuale è inefficiente e non riproducibile.

**Optuna** è un framework open-source per l'HPO automatizzato, scalabile e riproducibile, basato su ottimizzazione Bayesiana tramite l'algoritmo **TPE (Tree-structured Parzen Estimator)**.

**Come funziona il TPE:**
Anziché modellare direttamente $p(y|x)$ (la probabilità di un buon risultato data una configurazione di iperparametri), il TPE modella la densità inversa $p(x|y)$ suddividendo le osservazioni in due gruppi:
- $l(x)$: configurazioni migliori (top 15-25%)
- $g(x)$: restanti configurazioni

La **funzione di acquisizione** massimizza l'Expected Improvement (EI):

$$EI(x) \propto \frac{l(x)}{g(x)}$$

Il campionamento intelligente dirige la ricerca verso regioni statisticamente più promettenti, richiedendo molte meno prove rispetto alla Grid Search o alla Random Search.

In questo progetto, l'obiettivo primario era massimizzare il **Patient-Level Validation ROC-AUC**. Optuna usa il **pruning automatico** per terminare prematuramente le trial non promettenti (basandosi sulle metriche intermedie di validazione), risparmiando risorse computazionali significative.

#### Fine-Tuning Strategy: Gradual Unfreezing

Tutti i backbone (CNN e ViT) vengono inizializzati con i pesi pre-addestrati su **ImageNet-1K** (transfer learning). Il rischio principale è il **catastrophic forgetting**: se si aggiornano tutti i parametri contemporaneamente, le rappresentazioni generalizzabili apprese da ImageNet (bordi, texture, strutture geometriche) vengono rapidamente sovrascritte da feature specifiche del dominio medico, perdendo preziosa conoscenza.

Questo rischio è amplificato dalla grande differenza di dominio tra immagini naturali e mappe CMR.

**Soluzione: Gradual Unfreezing (sblocco graduale)**
- **Fase 1:** si addestra solo il Classifier Head (il backbone è congelato), adattando l'output del modello al task specifico.
- **Fase 2:** si sbloccano progressivamente i blocchi del backbone dal top (feature task-specific) al bottom (feature general-purpose):

$$n_{new}(t) = \left\lfloor B_{backbone} \cdot (t/T) \right\rfloor$$

dove $B_{backbone}$ è il numero totale di blocchi, $T$ il numero totale di epoche, $t$ l'epoca corrente.

**Layer-wise Learning Rates:**
- Classifier & Nuovi Blocchi Sbloccati → Learning Rate principale ($\alpha$)
- Blocchi precedentemente sbloccati → Learning Rate ridotto ($\alpha/10$) per preservare le rappresentazioni
- Blocchi congelati → esclusi dall'ottimizzazione

Questo approccio è **backbone-agnostic**: funziona allo stesso modo con CNN e ViT, rendendo il metodo modulare.

---

## 5. Drug Design nell'Era del Machine Learning e della Computational Intelligence

### 5.1 Cos'è un farmaco e come si sviluppa

#### Definizione di farmaco

Un farmaco è una sostanza chimica o biologica che, somministrata a un organismo vivente, interagisce con uno o più **bersagli biologici** (*biological targets*) producendo un effetto terapeutico, preventivo o diagnostico modificando funzioni fisiologiche o patologiche.

I principali bersagli biologici sono:
- **Proteine** (>90% dei casi): enzimi coinvolti nella segnalazione cellulare, recettori di membrana, canali ionici.
- **Acidi nucleici**: per la regolazione dell'espressione genica (farmaci antivirali, terapie epigenetiche).
- **Bersagli non convenzionali**: complessi macromolecolari.

#### Il processo di sviluppo di un farmaco

Lo sviluppo di un nuovo farmaco è un processo lungo (10-15 anni), costoso (miliardi di euro) e con alto tasso di fallimento (>90% dei candidati fallisce). Le fasi principali sono:

1. **Identificazione di molecole con attività biologica**
2. **Ottimizzazione dell'affinità e selettività per il bersaglio**
3. **Ottimizzazione delle proprietà ADMET** (Assorbimento, Distribuzione, Metabolismo, Escrezione, Tossicità)
4. **Integrazione dei requisiti chimici, biologici e farmacologici**

#### Approcci al Drug Design

| Approccio | Descrizione |
|-----------|-------------|
| **SBDD** (Structure-Based Drug Design) | Basato sulla struttura 3D del bersaglio (ottenuta tramite X-ray, NMR, cryo-EM, AlphaFold) |
| **LBDD** (Ligand-Based Drug Design) | Basato su ligandi noti e relazioni struttura-attività (QSAR, farmacoforici) |
| **De novo Drug Design** | Generazione de novo di molecole compatibili con il sito attivo |
| **FBDD** (Fragment-Based Drug Design) | Assemblaggio/ottimizzazione di piccoli frammenti a bassa affinità |

---

### 5.2 Lo spazio chimico e la sfida computazionale

Lo **spazio chimico** è l'insieme di tutti i possibili composti chimici. Più ampiamente viene esplorato, maggiore è la probabilità di trovare molecole biologicamente attive e candidati farmaci con le proprietà desiderate.

Ma la scala del problema è sconvolgente: **si stima che esistano circa $10^{60}$ molecole organicamente stabili**, rispetto ai $\sim 10^{24}$ stelle nell'universo osservabile. È impossibile sintetizzare e testare sperimentalmente anche solo una minuscola frazione di questo spazio.

Le tecniche computazionali permettono di **accelerare e ottimizzare l'esplorazione** di questo spazio immenso, guidando i chimici verso le regioni più promettenti senza dover sintetizzare ogni candidato.

---

### 5.3 Approcci al Drug Design computazionale

Il Computer-Aided Drug Design (CADD) è un campo che integra diverse tecniche computazionali per accelerare la scoperta di farmaci. I tre approcci descritti nel seminario sono:

1. Predire le reazioni metaboliche con un Molecular Transformer
2. Ottimizzare peptidi ciclici binders con un approccio guidato dalla struttura
3. Evolvere molecole con un algoritmo genetico

---

### 5.4 Predire le reazioni metaboliche con un Molecular Transformer

#### Il problema del metabolismo

Il **metabolismo** (principalmente epatico) trasforma i farmaci in metaboliti più idrofilici per facilitarne l'escrezione renale. Questa trasformazione influenza:
- L'effetto terapeutico del farmaco (un profarmaco inattivo può diventare attivo)
- La tossicità (un metabolita può essere più tossico del farmaco originale)
- Le interazioni farmaco-farmaco

Il metabolismo è quindi un argomento chiave in chimica medicinale, ma è stato storicamente **poco investigato** computazionalmente a causa della complessità delle reazioni enzimatiche coinvolte.

#### La notazione SMILES

Per rappresentare molecole chimiche in formato testo (necessario per i modelli di machine learning), si usa la notazione **SMILES** (Simplified Molecular-Input Line-Entry System). Ogni molecola viene codificata come una stringa ASCII che descrive la struttura molecolare:
- Atomi: simboli elementari (C=carbonio, N=azoto, O=ossigeno, ecc.)
- Legami: doppi (=), tripli (#), aromatici (minuscole)
- Rami: parentesi tonde
- Cicli: numeri che identificano gli atomi connessi

*Esempio:* L'aspirina (acido acetilsalicilico) → `CC(=O)Oc1cccc1C(=O)O`

#### Il Molecular Transformer

L'**approccio innovativo** consiste nel trattare la predizione del metabolismo come un **problema di traduzione automatica**: i substrati (molecole originali in SMILES) vengono "tradotti" nei metaboliti (prodotti del metabolismo, sempre in SMILES).

Questo sfrutta l'architettura **Transformer**, originariamente sviluppata per il Natural Language Processing (traduzione automatica di lingue). La struttura encoder-decoder permette di:
1. Codificare il substrato (in SMILES) come rappresentazione vettoriale contestuale
2. Decodificare il metabolita (in SMILES)

Il modello viene reso **consapevole della classe di reazione** (es. idrossilazione, glucuronidazione, ecc.), permettendo di predire sia il sito di metabolismo che il tipo di reazione.

**Dataset:** MetaQSAR, contenente coppie "substrato-metabolita" con la specificazione della classe di reazione coinvolta.

**Valutazione:** la correttezza delle previsioni viene misurata con la similarità Tanimoto su fingerprint molecolari ECFP:

$$S = \frac{c}{a + b - c}$$

dove $a$ e $b$ sono il numero di bit a 1 nelle fingerprint dei due composti, e $c$ il numero di bit a 1 in entrambe. $S=1$ indica molecole identiche, $S=0$ molecole senza feature in comune.

**Risultati:** il modello proposto supera significativamente MetaTrans (lo stato dell'arte precedente). La tabella seguente riporta il confronto completo tra i due modelli sviluppati e MetaTrans:

| Metrica | Modello 1 | Modello 2 | MetaTrans |
|---------|-----------|-----------|-----------|
| **Recall** | 99.6% | 99.6% | 57.9% |
| **Precision** | 54.5% | 62.4% | 53.4% |
| **F1 Score** | 70.5% | 76.8% | 55.5% |
| **Accuracy** | 54.1% | 62.3% | 38.8% |

Entrambi i modelli raggiungono un Recall quasi perfetto (99.6%), cruciale in un contesto medico dove i falsi negativi (metaboliti tossici non rilevati) sono pericolosi. Il Modello 2 supera il Modello 1 in Precision, F1 e Accuracy grazie a ottimizzazioni architetturali, mentre MetaTrans rimane significativamente indietro su tutte le metriche.

**Piano futuro:** automatizzare l'analisi delle predizioni per ridurre il costo del processo di validazione.

**Valore clinico:** permette di prioritizzare i candidati più promettenti, identificare potenziali tossicità e interazioni farmaco-farmaco, e comprendere i meccanismi biologici sottostanti.

---

### 5.5 Ottimizzazione di peptidi ciclici con Monte Carlo e Molecular Dynamics

#### Contesto biologico

Le **terapie basate su acidi nucleici** (es. siRNA, mRNA terapeutico) hanno il potenziale di trattare malattie genetiche, ma richiedono sistemi di rilascio (nanocarrier) che le proteggano dalla degradazione, superino le barriere cellulari e riducano l'immunogenicità.

I **peptidi ciclici** stanno emergendo come nanocarrier ideali per queste terapie. Il goal computazionale è progettare peptidi ciclici che si leghino a recettori tessuto-specifici per un rilascio mirato.

#### Caso di studio: target CD8

Il caso di studio concreto utilizza come target il recettore **CD8**, espresso sulle cellule T citotossiche del sistema immunitario — un bersaglio rilevante per le terapie basate su siRNA o mRNA che necessitano di essere internalizzate selettivamente in queste cellule.

| Target | Motivo noto (punto di partenza) | Fonte strutturale (PDB ID) |
|--------|--------------------------------|---------------------------|
| CD8 | DQTQDTE (dominio α3 di una molecola MHC di classe I umana) | 1AKJ |

Il motivo peptidico **DQTQDTE** è il punto di partenza dell'ottimizzazione: è un frammento della proteina MHC di classe I (complesso maggiore di istocompatibilità) che interagisce naturalmente con CD8. Il workflow computazionale (docking → Monte Carlo → MD) parte da derivati ciclizzati di questo motivo per esplorare lo spazio delle varianti con migliore affinità e proprietà strutturali.

#### Workflow di ottimizzazione

Il workflow integra tre metodi computazionali in sequenza:

**1. Molecular Docking**
Esplora le possibili orientazioni e conformazioni del peptide nel sito di legame del bersaglio, generate sulla base della complementarità geometrica e chimica. L'interazione viene valutata con **scoring functions** che approssimano l'affinità di legame (in kcal/mol). È un metodo rapido per generare e classificare ipotesi di legame, ma non considera la flessibilità dell'intero sistema né l'evoluzione temporale.

**2. Monte Carlo Algorithm con Criterio di Metropolis**
L'algoritmo di Monte Carlo esplora lo spazio delle sequenze peptidiche attraverso mutazioni casuali:
- **Atom-based mutation:** sostituisce un singolo atomo nella molecola
- **Fragment-based mutation:** sostituisce un frammento strutturale

Il **Criterio di Metropolis** governa l'accettazione o rifiuto di ogni mossa:
```python
Pacc = min(1, np.exp(-delta_e / T))  # T: temperatura corrente
if delta_e < 0 or random() <= Pacc:
    # Accetta la mutazione
```

Se la nuova soluzione è migliore ($\Delta E < 0$): sempre accettata.
Se la nuova soluzione è peggiore ($\Delta E > 0$): accettata con probabilità $P_{acc} = e^{-\Delta E / T}$.

Questo meccanismo di accettazione stocastica è cruciale: permette al sistema di **uscire dai minimi locali** (soluzioni sub-ottimali) accettando occasionalmente mosse "in salita" nell'energia. Man mano che la "temperatura" $T$ diminuisce (cooling schedule), diventa progressivamente più difficile uscire da un minimo locale — analogo al processo fisico del *simulated annealing*.

**3. Molecular Dynamics (MD)**
Le migliori soluzioni trovate dal Monte Carlo vengono validate con simulazioni di Molecular Dynamics. La MD simula il sistema in un ambiente realistico (solvente esplicito, ioni, temperatura e pressione controllate, forze fisiche reali). Testa se il modo di legame predetto dal docking è **stabile nel tempo**, non solo geometricamente plausibile.

I risultati del docking vengono filtrati osservando:
- Persistenza del legame
- Stabilità delle interazioni chiave
- Integrità strutturale del peptide ciclico

**Esempio di risultato:** le sequenze top identificate mostrano affinità di legame fino a -11.293 kcal/mol (TOP1), con interazioni chiave mediate da residui TRP2 e PHE10. La tabella seguente riporta le affinità delle quattro sequenze migliori:

| Peptide | Affinità (kcal/mol) | Interazioni chiave |
|---------|--------------------|--------------------|
| TOP1 | -11.293 | TRP2, PHE10 |
| TOP2 | -10.575 | PHE10, THR5 |
| TOP12 | -9.061 | — |
| TOP0 | -8.634 | — |

**Valore dell'approccio:** più veloce dei metodi puramente fisici, più affidabile degli algoritmi di Machine Learning puri.

---

### 5.6 Evoluzione di molecole con un Algoritmo Genetico

#### Algoritmi Genetici: concetti fondamentali

Gli **algoritmi genetici** (GA) sono metodi di ottimizzazione ispirata all'evoluzione darwiniana. Una popolazione di soluzioni candidate viene fatta evolvere attraverso operatori biologicamente ispirati:
- **Selezione:** le soluzioni migliori hanno più probabilità di "riprodursi"
- **Crossover (incrocio):** due soluzioni si combinano per generare una nuova soluzione figlia
- **Mutazione:** cambiamenti casuali nelle soluzioni figlie

In questo contesto, le "soluzioni" sono molecole rappresentate in formato SMILES, e la "fitness" di ogni molecola è una combinazione di:
1. **Affinità per il bersaglio** (quanto bene si lega al target)
2. **Sintetizzabilità** (quanto è facile da sintetizzare chimicamente)

#### Il database di partenza

Il GA parte da un database di molecole "lead-like" del database **ZINC15**, un repository pubblico di molecole acquistabili commercialmente. Questo garantisce che le molecole di partenza abbiano già proprietà farmacologiche di base (es. rispetto delle regole di Lipinski per la biodisponibilità orale).

#### Operatori di mutazione molecolare

A differenza dei GA classici che operano su stringhe di bit, qui le mutazioni devono avere senso chimico:
- **Atom-based mutation:** sostituzione di un singolo atomo (es. O→P)
  `CC(=O)Oc1cccc1C(=O)=O` → `CC(=O)Pc1ccnc1C(=O)=O`
- **Fragment-based mutation:** sostituzione di un frammento chimico (es. OH→P(=O)(O)O)
  Permette salti più grandi nello spazio chimico

#### Funzione di fitness: DrugClip e SA Scorer

**DrugClip** è un modello di deep learning basato su **apprendimento contrastivo**: mappa molecole e bersagli nello stesso spazio di embedding, in modo che molecole rilevanti per un certo bersaglio siano vicine nell'embedding space. Questa "distanza" viene usata come misura dell'affinità molecola-bersaglio.

Il **SA Scorer** (implementato in RDKit, la principale libreria open-source per la chemioinformatica) valuta quanto sia facile sintetizzare una molecola su una scala da 1 (facile) a 10 (difficile). Includere questo termine nella fitness impedisce al GA di evolvere molecole "chimicamente impossibili" ottimizzate solo per l'affinità computazionale.

#### Risultati preliminari

Dopo 200 generazioni, le molecole migliori mostrano:
- Affinità normalizzata: ~0.82 (molto alta)
- SA score: ~3.2 (moderatamente sintetizzabile)

La tabella seguente riporta le 10 molecole top generate dal GA (ordinate per affinità decrescente), con la loro rappresentazione SMILES, il valore di affinità normalizzato (DrugClip) e il SA score (RDKit):

| Gen. | SMILES | Affinità | SA Score |
|------|--------|----------|---------|
| 189 | `NC(=O)Oc1npccc1C(=O)Nc1ccc(C(N)=O)cn1` | 0.8243 | 3.23 |
| 93 | `NC(=O)OC1=NP=NC=C1C(=O)NC1=CC=C(C(N)=O)C=N1` | 0.8236 | 3.31 |
| 168 | `NC(=O)OC1=NP=NC=C1C(=O)Nc1ccc(C(N)=O)cn1` | 0.8236 | 3.31 |
| 78 | `NC(=O)C1=CN=C(NC(=O)C2=CC=PN=C2OC(=P)P)N=P1` | 0.8139 | 5.32 |
| 96 | `NC(=O)C1=PN=C(NC(=O)C2=CC=PN=C2OC(=O)Br)N=C1` | 0.8072 | 4.53 |
| 90 | `CC(=O)Oc1npccc1C(=O)Nc1ccc(C(N)=O)cn1` | 0.8038 | 3.14 |
| 145 | `CC(=O)OC1=NP=CC=C1C(=O)Nc1ccc(C(N)=O)cn1` | 0.8038 | 3.14 |
| 69 | `COOCOC1=NP=NC=C1C(=O)NC1=NP=C(C(N)=O)C=N1` | 0.7984 | 4.60 |
| 132 | `NC(=O)OC1=NP=CC=C1C(=O)NC1=CN=C(C(N)=O)C=N1` | 0.7597 | 3.46 |
| 36 | `NC(=O)C1=CN=C(NC(=O)C2=CC=PN=C2O)C=C1` | 0.7582 | 3.25 |

Le molecole top presentano affinità normalizzata elevata (>0.82) combinata con SA score relativamente basso (~3.2), indicando che il GA ha trovato un buon bilanciamento tra ottimizzazione dell'affinità e sintetizzabilità.

**Valore dell'approccio rispetto al docking tradizionale:**
- Più veloce del docking tradizionale (non richiede la struttura 3D del bersaglio)
- Abilita lo **screening virtuale ad alto throughput** di candidati promettenti
- Guida l'evoluzione verso molecole clinicamente rilevanti con le proprietà desiderate

**Piano futuro:** validare i candidati migliori con Molecular Dynamics, analisi dei contatti e scoring più sofisticato.

---

## 6. Connessioni Trasversali e Concetti Chiave per l'Esame

Rileggendo i tre seminari insieme, emergono connessioni profonde e temi ricorrenti che vale la pena sottolineare in vista dell'esame.

### Temi trasversali

**1. La robustezza come proprietà emergente e duale**
Sia nei sistemi biologici (Besozzi) che nei modelli di AI (Grazioso), la robustezza emerge dall'interazione tra componenti ed è sempre accompagnata da trade-off. Un modello AI robusto all'overfitting può essere fragile a distribution shift; un sistema biologico robusto alle mutazioni genetiche può essere vulnerabile a "dirottamenti" patologici (cancro).

**2. Il ruolo dei parametri: sensibilità vs. robustezza**
La PSA, la SA e la teoria delle biforcazioni (Besozzi) studiano come i parametri influenzano il sistema. Analogamente, nell'AI, l'ottimizzazione degli iperparametri (Optuna, Grazioso) e il fine-tuning graduale (gradual unfreezing) sono strategie per trovare configurazioni parametriche che garantiscono comportamenti desiderati. La "biforcazione" in un modello AI potrebbe essere la transizione da un training che generalizza a uno che overfita.

**3. Stochasticità e incertezza**
- Nelle simulazioni di sistemi biologici: simulazione stocastica (Gillespie) vs. deterministica (ODE)
- Nell'AI medica: soft labels vs. hard labels, calibrazione, Label Smoothing
- Nel drug design: Monte Carlo con criterio di Metropolis

In tutti i contesti, la gestione esplicita dell'incertezza (invece di ignorarla) porta a risultati più robusti e affidabili.

**4. Dall'ottimizzazione locale a quella globale**
- Parameter estimation: minimi locali vs. globale
- Monte Carlo / Simulated Annealing: accettazione probabilistica di mosse peggiori per evitare minimi locali
- Algoritmi genetici: esplorazione globale tramite diversità della popolazione
- Optuna/TPE: ottimizzazione Bayesiana intelligente dello spazio degli iperparametri

**5. Interpretabilità e spiegabilità**
La XAI (Grazioso) affronta il problema della "scatola nera" nei modelli deep learning. Analogamente, nel drug design computazionale (Multari) e nell'analisi dei sistemi biologici (Besozzi), la comprensione dei meccanismi — non solo la previsione dei risultati — è essenziale.

**6. Transfer e domain shift**
Il fine-tuning da ImageNet a CMR (Grazioso) è un esempio di transfer learning con domain shift. Anche nel drug design, i modelli addestrati su un tipo di molecole vengono adattati a nuove classi. La conoscenza pregressa (pesi pre-addestrati, database esistenti) accelera l'apprendimento.

---

### Glossario rapido dei termini chiave

| Termine | Definizione sintetica |
|---------|----------------------|
| **Attrattore** | Stato verso cui un sistema dinamico evolve nel lungo periodo |
| **Biforcazione** | Punto in cui una piccola variazione parametrica causa un cambiamento qualitativo nel sistema |
| **cAMP** | Adenosina monofosfato ciclica, secondo messaggero cellulare |
| **Calibrazione** | Corrispondenza tra le probabilità predette e le frequenze empiriche |
| **CMR** | Cardiovascular Magnetic Resonance, risonanza magnetica cardiovascolare |
| **Criterio di Metropolis** | Regola probabilistica di accettazione in Monte Carlo che permette di uscire da minimi locali |
| **ECV** | Extracellular Volume, misura del volume extracellulare nel miocardio |
| **EMA** | Exponential Moving Average, media mobile esponenziale |
| **Ensemble** | Combinazione di più modelli per ridurre varianza e migliorare robustezza |
| **Focal Loss** | Loss che down-pesa campioni "facili" per focalizzarsi su quelli difficili |
| **Grad-CAM** | Tecnica XAI che genera heatmap sulle regioni più influenti per la decisione |
| **Label Smoothing** | Tecnica di regolarizzazione che evita previsioni overconfident |
| **Mean Teacher** | Approccio SSL con studente (ottimizzato via SGD) e insegnante (EMA dei pesi) |
| **OAT** | One-Factor-At-a-Time, metodo di SA che varia un parametro alla volta |
| **Optuna/TPE** | Framework Bayesiano per l'ottimizzazione automatica degli iperparametri |
| **PSA** | Parameter Sweep Analysis, analisi sistematica dell'effetto dei parametri |
| **pyFUME** | Framework per sistemi di inferenza fuzzy intrinsecamente interpretabili (white-box AI) |
| **SA** | Sensitivity Analysis, analisi di sensibilità dei parametri di un modello |
| **SMILES** | Simplified Molecular-Input Line-Entry System, rappresentazione testuale di molecole |
| **SSL** | Semi-Supervised Learning, apprendimento con dati etichettati e non etichettati |
| **Temperature Scaling** | Calibrazione post-hoc che scala i logit per allineare confidenza e accuratezza |
| **Transfer Learning** | Utilizzo di pesi pre-addestrati su un task come inizializzazione per un task diverso |
| **ViT** | Vision Transformer, architettura transformer applicata alle immagini |
| **XAI** | Explainable AI, campo che studia come rendere i modelli AI comprensibili |

---

### Domande tipiche d'esame

1. Spiega la differenza tra robustezza e immutabilità in un sistema complesso. Fai esempi da almeno due contesti scientifici diversi.

2. Quali sono i meccanismi che garantiscono la robustezza di un sistema? Illustra il trade-off della robustezza nel contesto del cancro.

3. Descrivi la PSA-1D e la PSA-2D. Perché si usa il campionamento con sequenze di Sobol invece del campionamento pseudo-casuale?

4. Qual è la differenza tra SA locale e SA globale? Quando si usa ciascuna?

5. Descrivi le biforcazioni saddle-node, transcritica e pitchfork. Come cambia il numero e la stabilità dei punti fissi al variare del parametro di controllo?

6. Come si modella un pathway biologico e quali strumenti computazionali si usano per analizzarne la robustezza? Usa il pathway Ras/cAMP/PKA come esempio.

7. Spiega la Focal Loss e il Label Smoothing. Perché si combinano in un framework ibrido?

8. Descrivi il paradigma Mean Teacher per il Semi-Supervised Learning. Perché supera il self-training di base?

9. Qual è la differenza tra Explainability e Interpretability? Quali sono i limiti pratici della XAI nella clinica?

10. Descrive il workflow di ottimizzazione dei peptidi ciclici integrando Monte Carlo, Molecular Docking e Molecular Dynamics. Qual è il ruolo del Criterio di Metropolis?

11. Come funziona un Algoritmo Genetico per l'evoluzione di molecole farmaceutiche? Cosa misura il SA Scorer?

12. Come funziona il Molecular Transformer per la predizione del metabolismo? Perché il metabolismo è rilevante nel drug design?

---

*Dispensa generata a partire dai seminari presentati a Ca' Foscari University of Venice e all'Università di Milano-Bicocca nel periodo 2025-2026.*
