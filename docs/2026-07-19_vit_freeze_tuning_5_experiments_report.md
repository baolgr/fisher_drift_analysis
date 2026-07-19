# Réglage du gel Fisher sur ViT-small — rapport sur 5 expérimentations (2026-07-19)

Ce rapport analyse en profondeur les 5 dossiers suivants, tous le même modèle (ViT-small,
CIFAR-10, config cluster 40 epochs, `freeze_interval=7200`, seed=0, donc identiques jusqu'à
l'instant du gel) :

| Dossier | Rôle | `chunk_selection_metric` | `js_variance_lambda` |
|---|---|---|---|
| `runs/vit_small_nofreeze/` | témoin, aucun gel | — | — |
| `runs/vit_small_freeze/` | config validée à 18 epochs, réutilisée telle quelle à 40 epochs | `total_variation` | -1.0 |
| `runs/vit_small_freeze_fix_lambda13/` | tentative de correction n°1 | `total_variation` | **-1.3** |
| `runs/vit_small_freeze_fix_lambda16/` | tentative de correction n°2 | `total_variation` | **-1.6** |
| `runs/vit_small_freeze_fix_meanjs/` | tentative de correction n°3 | **`mean_js`** | -1.0 |

Aucun fichier n'a été modifié pour produire ce rapport — uniquement lecture de `summary.json`,
`train.log`, des configs `src/configs/config_vit_small_cluster_fix_*.yaml`, du code
(`src/fisher/trainer.py`, `src/fisher/fisher_core.py`, `src/utils/metrics.py`) et des PNG déjà
générés dans `metrics_plots/`.

## 0. Pourquoi ces 3 runs supplémentaires existent

`runs/vit_small_freeze/` (config validée à 18 epochs, portée telle quelle à 40 epochs) gèle en
réalité **90,9 % des paramètres** entraînables et ne conserve que **90,4 %** de l'accuracy du
témoin sans gel (voir `docs/2026-07-19_vit_cluster_run_curve_analysis.md` §6, qui a recalculé le
mécanisme de gel à partir du log SLURM récupéré) — beaucoup plus agressif que prévu, avec une
piste diagnostiquée : `total_variation` (la métrique qui décide quoi geler) est une **somme
cumulée** de petites variations mesurées toutes les 200 steps ; sur un entraînement 2× plus long
qu'à 18 epochs, elle accumule 2× plus de bruit de mesure, ce qui pénalise à tort les couches dont
les tenseurs sont petits. Les configs `fix_lambda13.yaml`/`fix_lambda16.yaml`/`fix_meanjs.yaml`
testent trois corrections indépendantes de cette hypothèse, une variable à la fois (même discipline
que le premier passage de réglage documenté dans `docs/fisheradaptune_freeze_experiments.md`) :

- **`fix_lambda13`/`fix_lambda16`** : baisser le seuil de gel (`λ` plus négatif) sans toucher à la
  métrique — teste si le problème est uniquement "le seuil est mal calé pour cet horizon".
- **`fix_meanjs`** : garder `λ=-1.0` mais remplacer la métrique cumulative par une **moyenne**
  (qui ne grossit pas avec le nombre de mesures) — teste directement l'hypothèse du biais de bruit
  cumulé.

## 1. Comment lire les graphiques

Cette section explique une fois pour toutes ce que chaque type de figure représente — le sens est
identique pour les 5 runs, seules les valeurs changent.

### 1.1 La dérive de Fisher (`fisher_drift.png`) — la métrique centrale

Pour chaque paramètre entraînable, le pipeline calcule une approximation de l'information de
Fisher (en gros : "à quel point ce poids est sensible / important pour la loss en ce moment").
Cette information est résumée sous forme d'un histogramme de valeurs. La **dérive de Fisher**
compare l'histogramme actuel à l'histogramme d'il y a quelques centaines de steps, via une
**distance de Jensen-Shannon** — une mesure de "à quel point deux distributions sont différentes",
qui vaut 0 si elles sont identiques et grandit à mesure qu'elles se ressemblent de moins en moins.

Point mathématique utile à savoir pour lire ces courbes : telle qu'implémentée ici (log népérien),
cette distance a un **plafond dur ≈ 0,83** (précisément √(ln 2)), atteint quand les deux
histogrammes ne se chevauchent plus du tout. **Ce n'est pas une limite numérique arbitraire ni un
bug** — c'est la valeur maximale mathématique de la formule. Presque toutes les courbes de ce
rapport finissent par se coller à ce plafond ; ça veut dire "ce groupe de poids a fini par changer
de régime de façon quasi complète depuis la référence", pas "erreur de calcul".

**En lien avec les expérimentations** : `fisher_drift.png` est tracé une ligne par couche
"repère" (7 couches choisies pour couvrir précoce/tardif et attention/MLP/plongement, cf.
`docs/plan.md`). C'est la seule figure où l'instant précis du gel (step 7200) est visible comme un
changement net de comportement pour les couches concernées. **Attention** : §5 ci-dessous montre
que cette figure peut donner une fausse impression de "saut brutal de dérive" pour une couche qui
vient d'être gelée — ce n'est pas un vrai sursaut de comportement, voir l'explication complète.

### 1.2 La magnitude de Fisher (`fisher_magnitude.png`)

Même information de Fisher que ci-dessus, mais sans la comparer à rien — juste sa valeur moyenne
absolue à cet instant. Sert à lever une ambiguïté : une dérive proche de 0 peut vouloir dire "cette
couche a une forte importance mais elle est stable" (bon candidat au gel) OU "cette couche n'a
jamais eu d'importance, il n'y a juste rien à mesurer" (couche quasi morte). Les deux donnent la
même courbe de dérive plate ; seule la magnitude les distingue. Dans les 5 runs, `patch_embed.proj`
a une magnitude nettement plus basse (~0,044) que les autres couches (~0,13-0,20) — signal
architectural stable (c'est une conv, pas un Linear), sans lien avec le gel.

### 1.3 Norme du gradient (`grad_norm.png`)

Grandeur d'optimisation classique, indépendante de tout calcul Fisher : la norme du gradient reçu
par la couche à cet instant précis (juste après le backward, avant que l'optimiseur ne bouge les
poids — voir `CLAUDE.md` sur pourquoi cet instant précis compte). Sert de repère de sanité : si le
gradient est non nul mais la dérive Fisher est plate, ça mérite une explication ; si le gradient
lui-même est nul, ce n'est pas surprenant. **Dans ce rapport, c'est la métrique qui confirme le
plus proprement quelle couche est gelée** : une couche gelée a un gradient masqué à zéro par
construction (le mécanisme de gel force `param.grad = 0` sur les éléments gelés), donc sa courbe
tombe et reste plate à 0 exactement au step du gel, sans ambiguïté.

### 1.4 Mise à jour relative des poids (`relative_update.png`)

`‖Δw‖ / ‖w‖` — de combien les poids ont *réellement* bougé entre deux mesures, après le pas
AdamW complet (donc après weight decay etc.), rapporté à la taille des poids eux-mêmes. C'est la
métrique la plus directe pour savoir "cette couche apprend-elle encore, concrètement" : contrairement
à `fisher_drift`, elle est calculée directement sur les poids du module (pas sur une moyenne de
"chunks" internes du trainer), donc **elle n'a pas l'ambiguïté décrite en §5** — une couche
vraiment gelée y tombe proprement à 0 et y reste. C'est la métrique utilisée dans ce rapport pour
déterminer, sans ambiguïté, quelle couche est gelée dans chaque run (voir tableau §4).

### 1.5 Les 3 graphiques "population complète" (heatmap / tiers / violon)

Les 4 courbes ci-dessus ne suivent que 7 couches "repères" sur les ~90-160 tenseurs entraînables
du modèle (choisies pour rester lisibles). Ces 3 figures, elles, résument **toute la population**
de chunks Fisher suivis par le trainer (chaque poids est en fait découpé en 4 sous-blocs de lignes,
`fisher_slice_blocks=4`, donc plusieurs centaines de "chunks" au total) :

- **`chunk_drift_heatmap.png`** : une ligne par couche, une colonne par sous-bloc, couleur = dérive
  finale (mêmes 0 à ~0,83 qu'en §1.1). Les lignes sont triées de la dérive la plus forte (bas,
  bleu foncé) à la plus faible (haut, clair). La bande latérale colorée indique le groupe
  structurel (Plongement / Blocs précoces / Blocs tardifs / Tête). La barre de droite montre
  l'écart interne à chaque couche entre son sous-bloc le plus et le moins dérivant.
- **`chunk_drift_trajectories_by_tier.png`** : les chunks sont classés en 3 groupes selon leur
  dérive finale (Haut/Moyen/Bas, 6/4/6 chunks chacun) et on trace leur trajectoire dans le temps.
  Sert à voir si la structure "3 paliers" est stable et interprétable, ou si c'est du bruit.
- **`chunk_drift_violin_by_param_type.png`** : la dérive finale de *chaque* chunk encore suivi à
  la fin de l'entraînement, regroupée par type de paramètre (Norm = LayerNorm, Attn = matrices
  Q/K/V, MLP-FC = les deux couches du MLP, Other = plongement + tête). La forme "violon" montre la
  distribution complète (pas juste une moyenne), et **le nombre `n=` sous chaque groupe est la
  donnée la plus utile de cette figure pour ce rapport** : c'est directement le nombre de chunks de
  ce type qui **survivent** au gel (un chunk gelé est retiré du suivi, donc disparaît de ce
  graphique — voir §6, ce comptage sert de preuve directe de "qui a été gelé").

### 1.6 Le mécanisme de gel lui-même (pas un graphique, mais nécessaire pour tout lire)

Au step 7200, chaque chunk encore entraînable reçoit un score (`total_variation` ou `mean_js`
selon la config). Le seuil de gel est `seuil = moyenne(scores) + λ × écart-type(scores)`. Les
chunks dont le score est **en dessous** du seuil sont gelés (considérés "stables" → plus rien à
apprendre) ; ceux au-dessus restent entraînables. `λ` est toujours négatif dans ces 5 runs : plus
il est négatif, plus le seuil descend, donc **moins on gèle**. C'est un déclencheur unique (pas
progressif) : tout se joue en un seul step, ce qui explique pourquoi toutes les courbes ci-dessus
montrent une transition nette à un instant précis plutôt qu'une évolution graduelle.

## 2. Résultats chiffrés

| Run | métrique | λ | params entraînables | % conservé | test accuracy | accuracy relative* |
|---|---|---|---|---|---|---|
| `nofreeze` | — | — | 2 680 906 / 2 693 578 | 99,5 % | **0,7273** | 100 % (référence) |
| `freeze` (baseline) | total_variation | -1,0 | 245 818 | **9,1 %** | 0,6574 | 90,4 % |
| `freeze_fix_lambda13` | total_variation | -1,3 | 1 796 170 | **66,7 %** | **0,7212** | **99,2 %** |
| `freeze_fix_lambda16` | total_variation | -1,6 | 2 533 450 | **94,1 %** | **0,7213** | **99,2 %** |
| `freeze_fix_meanjs` | mean_js | -1,0 | 1 316 362 | **48,9 %** | 0,7095 | 97,6 % |

*accuracy relative = test accuracy du run / test accuracy de `nofreeze`.

*(`nofreeze` n'a pas 100 % des params entraînables : un mécanisme séparé, indépendant de
`js_variance_lambda`, gèle dès le step 0 les 2 chunks à Fisher exactement nul dans les 5 runs —
un garde-fou générique du trainer, sans rapport avec le réglage étudié ici.)*

**Le résultat le plus net de ce rapport** : les trois tentatives de correction fonctionnent
toutes, mais très différemment :

- `lambda13` et `lambda16` **quasi éliminent la perte d'accuracy** (99,2 % relative dans les deux
  cas — un écart de 0,0001 entre eux, probablement du bruit à seed unique, voir §7). Fait notable :
  geler 33 % des paramètres (`lambda13`) coûte la même accuracy que n'en geler que 6 %
  (`lambda16`) — signe que la tranche de paramètres gelée en plus par `lambda13` était réellement
  superflue, exactement l'hypothèse que le mécanisme FisherAdapTune est censé vérifier.
- `meanjs` retrouve un compromis différent : gèle la moitié du réseau (48,9 % conservé) pour un
  coût plus visible mais toujours modéré (2,4 points d'accuracy, 97,6 % relatif).

### Le choc de gel, mesuré sur l'accuracy de validation

Les 4 runs à gel partagent le même entraînement jusqu'au step 7200 (epoch 20, accuracy de
validation = 0,6678 dans les 4 cas). L'écart d'accuracy entre l'epoch juste avant et juste après
le gel montre un choc proportionnel à l'agressivité du gel :

| Run | accuracy epoch 20 | accuracy epoch 21 | choc | % params gelés | Early stopping ? |
|---|---|---|---|---|---|
| `freeze` (baseline) | 0,6678 | 0,4951 | **-17,3 points** | 90,9 % | oui, epoch 32 |
| `freeze_fix_meanjs` | 0,6678 | 0,6088 | -5,9 points | 51,1 % | non, 40 epochs complétées |
| `freeze_fix_lambda13` | 0,6678 | 0,6543 | -1,4 points | 33,3 % | non, 40 epochs complétées |
| `freeze_fix_lambda16` | 0,6678 | 0,6592 | -0,9 point | 5,9 % | non, 40 epochs complétées |

Relation quasi monotone entre "fraction du réseau gelée d'un coup" et "taille du choc" — cohérent
avec l'idée qu'un gel simultané massif force une redistribution brutale du signal de gradient sur
les quelques paramètres restants. Autre conséquence concrète : le choc du run `freeze` baseline
est assez sévère pour déclencher l'arrêt anticipé (patience=12) avant la fin des 40 epochs
prévues, alors que les 3 corrections laissent l'entraînement se dérouler jusqu'au bout — un budget
d'entraînement gaspillé en pure récupération post-choc plutôt qu'en progrès, en plus du coût
d'accuracy final.

## 3. Quelles couches sont gelées, selon quelle stratégie

En croisant `relative_update.png` et `grad_norm.png` (les deux métriques qui montrent sans
ambiguïté "cette couche a-t-elle encore bougé après le step 7200", voir §1.3-1.4) sur les 7
couches repères :

| Couche repère | `freeze` (λ=-1,0, tv) | `lambda13` (λ=-1,3, tv) | `lambda16` (λ=-1,6, tv) | `meanjs` (λ=-1,0, mean_js) |
|---|---|---|---|---|
| `patch_embed.proj` | active | active | active | active |
| `blocks.0.attn.qkv` | **gelée** | active | active | **gelée** |
| `blocks.0.mlp.fc1` | **gelée** | active | active | active |
| `blocks.2.attn.qkv` | **gelée** | active | active | **gelée** |
| `blocks.5.attn.qkv` | **gelée** | active | active | active |
| `blocks.5.mlp.fc2` | **gelée** | **gelée** | **gelée** | **gelée** |
| `head` | **gelée** | active | active | active |

Deux observations exploitables pour la réunion :

1. **`blocks.5.mlp.fc2` est systématiquement le premier tenseur touché**, y compris dans
   `lambda16` où seuls 4 chunks sur 408 sont gelés dans tout le réseau. C'est cohérent avec un
   signal réel et reproductible plutôt qu'un hasard : ce tenseur particulier atteint un score de
   dérive bas plus tôt/plus systématiquement que les autres.
2. **`meanjs` est la seule config qui gèle une couche d'attention "précoce" (`blocks.0`/`.2`) tout
   en épargnant l'attention "tardive" (`blocks.5`)** — exactement l'axe précoce-vs-tardif que
   l'encadrant demande explicitement d'étudier (`docs/plan.md`, point 7). Les deux configs
   `total_variation` (`lambda13`/`lambda16`), elles, ne touchent **jamais** à l'attention à ces
   réglages (voir aussi le violon §6) : elles ne peuvent tout simplement pas produire ce genre de
   résultat, pas parce que l'attention y est "plus stable" en vérité, mais parce que la métrique
   cumulative ne laisse jamais les chunks d'attention descendre sous le seuil dans la plage de λ
   testée ici.

## 4. Une découverte méthodologique : le "saut" de dérive Fisher, expliqué

`docs/2026-07-19_vit_cluster_run_curve_analysis.md` §3 avait repéré, sans trancher, un phénomène
troublant : certaines couches *non gelées* affichent un saut brutal de `fisher_drift` exactement au
step du gel — alors que rien ne devrait changer dans leur propre comportement. En comparant les 4
runs à gel, ce rapport peut maintenant l'expliquer précisément (code : `src/utils/metrics.py:66-71`
et `src/fisher/fisher_core.py:452-473`).

**Le mécanisme** : `fisher_drift` pour une couche repère n'est pas mesuré sur "la couche" comme un
bloc — c'est la **moyenne** des scores de ses sous-blocs internes (les "chunks", jusqu'à 4 par
poids + biais). Quand un chunk est gelé, il est **supprimé** du suivi interne du trainer (pas mis à
zéro — supprimé). Deux cas de figure :

- **Gel complet** (tous les chunks d'une couche repère sont gelés en même temps) : la moyenne n'a
  plus rien à moyenner → elle retombe explicitement à 0 par construction du code. C'est ce qui a
  été vu sur les couches gelées de ResNet50 dans le rapport précédent
  (`docs/2026-07-19_resnet50_cluster_run_curve_analysis.md` §3).
- **Gel partiel** (certains sous-blocs d'une couche repère franchissent le seuil, d'autres non,
  c'est le cas de `blocks.5.mlp.fc2` dans `lambda13`/`lambda16`/`meanjs`) : la moyenne continue,
  mais sur un groupe de survivants plus petit — et par construction du mécanisme de sélection
  (§1.6), les chunks gelés sont *toujours* ceux qui avaient le score **le plus bas**. Les retirer
  de la moyenne pousse donc mécaniquement la moyenne des survivants **vers le haut**. C'est un
  effet de recomposition du groupe moyenné, pas une accélération réelle de la dérive des poids
  restants. Le même mécanisme touche `fisher_magnitude` (visible sur
  `runs/vit_small_freeze_fix_lambda13/metrics_plots/fisher_magnitude.png` : `blocks.5.mlp.fc2`
  saute de 0,127 à 0,198 au step du gel, exactement au même instant).

**Conséquence pratique** : sur les figures `fisher_drift`/`fisher_magnitude` curées (7 couches),
un saut juste après le step de gel ne doit **pas** être lu comme "cette couche s'est mise à changer
plus vite" — c'est un artefact de la façon dont la moyenne est recalculée. `relative_update` et
`grad_norm` n'ont pas ce problème (ils ne moyennent pas sur des sous-blocs internes) et restent la
source fiable pour savoir ce qui est réellement gelé (tableau §3). Les graphiques "population
complète" (§1.5) travaillent directement au niveau du chunk et n'ont pas non plus ce problème.

## 5. Ce que le violon "population complète" confirme, chiffres à l'appui

Les `n=` du violon par type de paramètre (§1.5) donnent, pour chaque run, le nombre de chunks
encore suivis en fin d'entraînement par catégorie — donc, par différence avec `nofreeze`
(Norm=104, Attn=192, MLP-FC=96, Other=16, total 408), le nombre de chunks gelés par catégorie :

| Run | Norm gelés | Attn gelés | MLP-FC gelés | Other gelés | Total gelé |
|---|---|---|---|---|---|
| `freeze` (baseline) | 3 | 72 | 48 | 4 | 127 |
| `lambda13` | 0 | **0** | 24 | 0 | 24 |
| `lambda16` | 0 | **0** | 4 | 0 | 4 |
| `meanjs` | 12 | 36 | 28 | 0 | 76 |

Ce tableau confirme numériquement §3 : `lambda13`/`lambda16` (métrique `total_variation`) ne
touchent **jamais** à Norm ni Attn dans cette plage de λ — tout le gel se concentre sur MLP-FC.
`meanjs` est la seule config qui distribue le gel sur les trois catégories à la fois. C'est une
différence qualitative entre les deux stratégies de correction, pas seulement une différence de
quantité totale gelée.

## 6. Conclusions pour la réunion du 21 juillet

1. **Le bug de sur-gel identifié pour la config à 40 epochs est confirmé corrigé, par deux voies
   indépendantes.** `lambda13` (recommandé, voir point 2) et `lambda16` ramènent l'accuracy
   relative de 90,4 % à 99,2 %.
2. **`lambda13` (λ=-1,3, `total_variation`) est le meilleur compromis des 4 configs testées** :
   même accuracy que `lambda16` (99,2 % relatif) mais en gelant réellement 33,3 % des paramètres
   au lieu de 5,9 % — `lambda16` gèle trop peu pour représenter un résultat intéressant pour
   FisherAdapTune (démontrer qu'on peut geler *sans perte*, pas qu'on peut à peine geler).
3. **`meanjs` reste intéressant comme deuxième point de comparaison**, pas comme remplacement :
   moins bon compromis accuracy/rétention que `lambda13` sur ce run précis, mais c'est la seule
   config qui montre un gel sensible aux couches d'attention et à l'axe précoce/tardif — un
   résultat qualitativement plus riche à présenter si la question porte sur "quelles couches sont
   gelées" plutôt que sur l'accuracy pure.
4. **Le choc de gel (§2) est un phénomène réel et mesurable**, proportionnel à l'agressivité du
   gel — argument supplémentaire en faveur d'un seuil moins agressif (`lambda13`/`lambda16`) :
   au-delà de l'accuracy finale, ces configs laissent l'entraînement se dérouler sans interruption
   ni gaspillage de budget en récupération post-choc.
5. **Point de prudence à mentionner si ces figures sont montrées telles quelles** : le "saut" de
   `fisher_drift`/`fisher_magnitude` juste après un step de gel (visible sur `blocks.5.mlp.fc2`
   dans les 4 runs à gel) est un artefact de moyenne, pas un vrai changement de comportement — voir
   §4. Préférer `relative_update`/`grad_norm` ou les graphiques "population complète" pour toute
   affirmation sur "qu'est-ce qui est gelé".

## 7. Pistes d'amélioration explorées et à explorer

- **Combiner les deux corrections** (`chunk_selection_metric="mean_js"` avec un λ moins agressif
  que -1,0, ex. -1,3) : piste déjà notée dans l'en-tête de
  `config_vit_small_cluster_fix_lambda13.yaml` mais jamais lancée. Intérêt : `meanjs` seul touche
  déjà l'attention de façon crédible (§3) mais coûte 2,4 points d'accuracy à λ=-1,0 ; un λ plus
  permissif pourrait retrouver une accuracy proche de `lambda13` tout en gardant la sélectivité
  par type de paramètre plus réaliste de `mean_js`. C'est la piste la plus prometteuse à lancer en
  premier si un créneau de calcul reste disponible avant le 21.
- **Répliquer `lambda13` sur une deuxième seed.** `lambda13` (0,7212) et `lambda16` (0,7213) sont
  quasi indiscernables malgré un facteur ~3 sur la fraction de paramètres gelés — c'est le résultat
  le plus intéressant de ce rapport (« on peut geler un tiers du réseau gratuitement ») mais il
  repose sur un seul run par config (seed=0 partout). Une deuxième seed sur `lambda13` seul (le
  point le plus utile à défendre) suffirait à vérifier que l'écart avec `lambda16` n'est pas du
  bruit d'entraînement.
- **Corriger l'artefact de moyenne décrit en §4** dans `src/utils/metrics.py`/`src/utils/plotting.py`
  avant de montrer les courbes curées telles quelles à un public qui ne connaît pas ce détail —
  par exemple en excluant purement les couches entièrement gelées du tracé après leur gel (au lieu
  de retomber sur 0), ou en ajoutant un marqueur visuel (ligne verticale, changement de style de
  trait) au step de gel sur `fisher_drift.png`/`fisher_magnitude.png`. Non fait ici (pas de
  modification de fichier demandée pour ce rapport), mais peu coûteux si le temps le permet.
- **Appliquer le même sweep λ/métrique à ResNet50.** Le run ResNet50 déjà analysé
  (`docs/2026-07-19_resnet50_cluster_run_curve_analysis.md`) utilise λ=-1,0/`total_variation` sans
  jamais avoir été comparé à d'autres réglages — il se trouve dans une zone d'accuracy déjà très
  favorable (99,67 % relatif) mais rien ne garantit que λ=-1,0 y soit optimal plutôt que
  simplement "pas assez agressif pour faire mal" ; un sweep équivalent donnerait une méthodologie
  cohérente entre les deux modèles avant de comparer leurs résultats l'un à l'autre.
- **Isoler et présenter séparément le résultat "attention précoce vs tardive" de `meanjs`**
  (§3, point 2) : c'est la réponse la plus directe aux points 4 et 7 de la demande de l'encadrant
  dans `docs/plan.md`, mais elle est actuellement noyée dans un tableau croisé à 7 lignes. Un
  graphique dédié (dérive finale par indice de bloc, pour les seuls chunks d'attention) rendrait
  ce résultat immédiatement lisible en réunion.

## 8. Limites de ce rapport

Toutes les lectures de `relative_update`/`grad_norm` (§3) sont des lectures visuelles de courbes
PNG, pas une extraction programmatique des masques de gel réels — cohérentes entre elles (deux
métriques indépendantes, même conclusion) et avec les comptages du violon (§6), mais pas une preuve
au sens strict d'un dump direct de `trainer._trainable_chunks`. L'explication du "saut" (§4) est
bien ancrée dans le code lu (`metrics.py`, `fisher_core.py`), mais n'a pas été vérifiée en
instrumentant un run réel (ex. logguer `len(js_values)` à chaque step) — recommandé si ce point
précis devient central dans une publication ou une décision d'implémentation.
