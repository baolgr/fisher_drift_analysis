# Analyse du batch SLURM de la nuit du 19 au 20 juillet 2026

Les 12 jobs soumis hier soir (6 relances + 6 nouveaux, voir `slurm/README.md`) ont tous terminé.
Cette note analyse les résultats — `summary.json`, `train.log`, et les nouveaux plots
(marqueur de gel + overlay accuracy) produits par le code mis à jour hier. Aucun fichier de run
modifié ici, uniquement lecture.

## 0. Anomalie à signaler avant toute chose

`runs/vit_small_freeze_fix_lambda16/` a disparu du disque — ni relancé (exclu volontairement,
voir la session précédente) ni écrasé par un des 12 jobs. Il est dans la corbeille macOS
(`~/Library/Mobile Documents/.Trash/vit_small_freeze_fix_lambda16`), donc supprimé manuellement
en dehors de cette session, pas par un des scripts. Récupérable si besoin (glisser hors de la
corbeille) — je n'y ai pas touché. Conséquence directe : le sweep ViT ci-dessous et
`runs/sweep_summaries/vit_lambda_metric_sweep.png` ne comportent plus ce point (5 configs de gel
au lieu de 6). Les chiffres de `lambda16` restent ceux déjà cités dans
`docs/2026-07-19_vit_freeze_tuning_5_experiments_report.md` (0.7213, 94.1% conservé) si besoin de
les reciter.

## 1. Déterminisme confirmé

Les 6 runs relancés avec le nom d'origine (`vit_small_nofreeze`, `vit_small_freeze`,
`vit_small_freeze_fix_lambda13`, `vit_small_freeze_fix_meanjs`, `resnet50_nofreeze`,
`resnet50_freeze`) reproduisent des chiffres **identiques au bit** à ceux déjà cités dans les
rapports du 19 juillet (ex. `vit_small_freeze` : 245 818 params entraînables, accuracy
0.6574367088607594, exactement). Même seed, même hardware, même résultat — confirme qu'aucune
régression n'a été introduite par les changements de plotting (qui ne touchent que la sortie
graphique, jamais l'entraînement lui-même).

## 2. Sweep ResNet50 — un vrai point de rupture existe, juste plus loin qu'à λ=-1,0

| Run | métrique | λ | params entraînables | % conservé | test accuracy | accuracy relative* |
|---|---|---|---|---|---|---|
| `resnet50_nofreeze` | — | — | 23 520 842 | 100,0 % | 0,9027 | 100,0 % (réf.) |
| `resnet50_freeze` (baseline) | total_variation | -1,0 | 7 717 578 | 32,8 % | 0,8997 | 99,67 % |
| `resnet50_freeze_sweep_lambda07` | total_variation | -0,7 | 3 454 858 | 14,7 % | 0,8824 | 97,75 % |
| `resnet50_freeze_sweep_lambda04` | total_variation | **-0,4** | **112 074** | **0,5 %** | **0,3989** | **44,19 %** |
| `resnet50_freeze_sweep_lambda13` | total_variation | -1,3 | 16 915 018 | 71,9 % | **0,9119** | **101,02 %** |
| `resnet50_freeze_sweep_meanjs` | mean_js | -1,0 | 6 406 474 | 27,2 % | 0,9021 | 99,93 % |

*relative au `nofreeze` de ce même batch.

Figure : `runs/sweep_summaries/resnet50_lambda_metric_sweep.png`.

**Le résultat le plus important de cette nuit** : contrairement à ce que le run baseline
(`resnet50_freeze`, 99,67 % relatif) laissait penser, ResNet50 a bien un point de rupture
catastrophique comme le ViT — juste situé beaucoup plus loin dans la plage de λ testée
initialement. Entre λ=-0,7 (97,75 % relatif, encore raisonnable) et λ=-0,4 (44,19 % relatif, quasi
destruction du réseau), l'accuracy s'effondre en une seule marche : de 14,7 % à seulement **0,5 %**
de paramètres conservés (112 074 sur 23,5M). Log du gel à λ=-0,4 :

```
[Freeze] Step 7200: variation mean=0.5611 std=0.2529 λ=-0.400 threshold=0.4600 | frozen=250 remaining=394
[Freeze] Step 7200: trainable tensors=99 chunks=394 params=112,074
```

Le choc de gel mesuré sur l'accuracy de validation confirme et quantifie l'effondrement :

| Run | acc. epoch 20 | acc. epoch 21 | choc | acc. epoch 22 (récup. partielle) |
|---|---|---|---|---|
| `sweep_lambda13` | 0,8936 | 0,8719 | -2,2 pts | 0,8768 |
| `resnet50_freeze` (baseline) | 0,8936 | 0,7578 | -13,6 pts | 0,8629 |
| `sweep_meanjs` | 0,8936 | 0,7713 | -12,2 pts | 0,8514 |
| `sweep_lambda07` | 0,8936 | 0,4957 | -39,8 pts | 0,7031 |
| `sweep_lambda04` | 0,8936 | **0,1193** | **-77,4 pts** | 0,1709 (pas de récupération) |

**Même biais structurel déjà documenté** (`docs/2026-07-19_resnet50_cluster_run_curve_analysis.md`
§3, mémoire `fisheradaptune_freeze_collapse_bug`) : le pool de chunks entraînables restants
rétrécit à la fois en nombre *et* en taille moyenne à mesure que λ devient moins négatif —
16 915 018/559 chunks ≈ 30 260 params/chunk à λ=-1,3, contre 112 074/394 ≈ 284 params/chunk à
λ=-0,4. Les gros tenseurs (convolutions) sont gelés en priorité ; ce qui reste à λ=-0,4 n'est plus
qu'un résidu de petits paramètres BatchNorm — le réseau perd l'essentiel de sa capacité
convolutive d'un coup. Confirme que ce n'est pas un artefact ViT-spécifique mais une propriété
générale de `total_variation` comme critère de sélection, qui se manifeste simplement à un point
différent de la plage de λ selon l'architecture.

**Deux résultats positifs, à nuancer avant de les présenter tels quels** :
- `sweep_lambda13` (λ=-1,3) **dépasse légèrement le nofreeze** (0,9119 vs 0,9027, +1,02 % relatif)
  en conservant 71,9 % des paramètres — plausible comme effet de régularisation léger (geler des
  chunks stabilisés réduit le surapprentissage résiduel), mais c'est un seul seed : à traiter comme
  "gel n'est pas pire, et peut légèrement aider" plutôt qu'un gain confirmé statistiquement.
- `sweep_meanjs` (mean_js, λ=-1,0) égale quasiment le nofreeze (99,93 % relatif) en gelant
  **davantage** que le baseline `total_variation` (27,2 % conservé contre 32,8 %) — un point de
  compromis légèrement meilleur que le baseline actuel à λ égal.

## 3. Probes ViT — la piste "combiner les deux corrections" ne tient qu'à moitié

| Run | métrique | λ | params entraînables | % conservé | test accuracy | accuracy relative* |
|---|---|---|---|---|---|---|
| `vit_small_nofreeze` | — | — | 2 680 906 | 99,5 % | 0,7273 | 100,0 % (réf.) |
| `vit_small_freeze_fix_meanjs_lambda07` | mean_js | **-0,7** | **23 386** | **0,9 %** | **0,5206** | **71,58 %** |
| `vit_small_freeze_fix_meanjs` (déjà connu) | mean_js | -1,0 | 1 316 362 | 48,9 % | 0,7095 | 97,55 % |
| `vit_small_freeze_fix_lambda13` (déjà connu) | total_variation | -1,3 | 1 796 170 | 66,7 % | 0,7212 | 99,17 % |
| `vit_small_freeze_fix_meanjs_lambda13` (**combiné**) | mean_js | -1,3 | 1 943 050 | 72,1 % | 0,7209 | 99,13 % |

Figure : `runs/sweep_summaries/vit_lambda_metric_sweep.png`.

**`meanjs_lambda07` confirme que `mean_js` a lui aussi son propre point de rupture**, symétrique à
celui de `total_variation` : à λ=-0,7 (plus agressif que le -1,0 déjà validé), le réseau
s'effondre à 0,9 % de paramètres conservés (23 386 sur 2,68M), choc de -43,4 points d'accuracy
(0,6678 → 0,2340 entre l'epoch 20 et 21), sans récupération franche (0,309 à l'epoch 22, test final
71,6 % relatif). `mean_js` **n'évite donc pas** le mécanisme d'effondrement catégorique — il ne
fait que déplacer le seuil où il se produit, exactement comme observé côté ResNet50 (§2).

**Le résultat qualitativement le plus important de cette nuit** concerne `meanjs_lambda13` (la
correction combinée, la piste jugée la plus prometteuse du rapport précédent §7) : elle atteint
bien une accuracy quasi identique à `lambda13` seul (99,13 % vs 99,17 % relatif) — mais **perd la
propriété qui rendait `mean_js` intéressant en premier lieu**. Comptage des chunks gelés par type
de paramètre (`chunk_drift_violin_by_param_type.png`, population totale 408 chunks : Norm=104,
Attn=192, MLP-FC=96, Other=16) :

| Run | Norm gelés | Attn gelés | MLP-FC gelés | Other gelés | Total gelé |
|---|---|---|---|---|---|
| `meanjs` (λ=-1,0) | 12 | **36** | 28 | 0 | 76 |
| `meanjs_lambda13` (combiné, λ=-1,3) | 12 | **0** | 20 | 0 | 32 |

En assouplissant λ pour retrouver l'accuracy de `lambda13`, le gel de l'attention **disparaît
complètement** (36 → 0 chunks) — alors que le gel des chunks Norm reste identique (12 dans les deux
cas). Autrement dit : la sélectivité par type de paramètre de `mean_js` (le résultat qualitativement
riche du rapport précédent, §3 point 2 et §6 point 3) et la bonne accuracy de `lambda13` sont
**mutuellement exclusives dans la plage de λ testée** — on ne peut pas avoir les deux en même temps
avec ce réglage. C'est une piste explicitement infirmée, pas juste "non concluante" : à documenter
comme telle plutôt que reformulée en tentative supplémentaire.

## 4. Les nouveaux plots (marqueur de gel + overlay accuracy) confirmés en conditions réelles

Vérifié visuellement sur plusieurs runs (`fisher_drift.png` de chaque dossier) : la ligne verticale
pointillée "freeze" et la courbe d'accuracy (axe de droite, noir, tirets) s'affichent correctement
sur les 12 runs. Sur `resnet50_freeze_sweep_lambda04` (le cas d'effondrement), la figure rend
immédiatement visible ce que le tableau §2 ne montre qu'en chiffres : la courbe d'accuracy chute
à la verticale du marqueur de gel et ne remonte que très lentement, pendant que `fc` (non gelée)
montre le saut d'artefact de moyenne décrit en §4 du rapport précédent (0,49 → 0,81 juste après le
marqueur) — exactement le genre de lecture erronée que le marqueur est censé prévenir.

Chaque run a bien écrit `metrics_plots/history.json` (vérifié sur `resnet50_freeze_sweep_lambda13`)
— `scripts/regenerate_plots.py` fonctionnera sur ces 12 runs pour tout futur changement de
plotting, sans devoir les relancer une troisième fois.

## 5. Bottom line pour la suite

1. **`total_variation` et `mean_js` ont chacun un vrai point de rupture catastrophique**, pas
   spécifique à un modèle — juste situé à un λ différent selon l'architecture/métrique
   (ViT+total_variation : déjà franchi à λ=-1,0 ; ResNet50+total_variation : entre -0,7 et -0,4 ;
   ViT+mean_js : entre -1,0 et -0,7). Argument méthodologique fort pour la réunion : ne jamais
   déployer une nouvelle combinaison modèle/métrique/λ sans un point de comparaison de part et
   d'autre, le comportement n'est pas monotone-doux, c'est un seuil dur.
2. **ResNet50 accepte bien plus de gel que le baseline λ=-1,0 ne le suggérait** :
   `sweep_lambda13` conserve l'accuracy (et la dépasse légèrement) en gelant 28,1 % de plus que le
   baseline (71,9 % conservé) — piste concrète à mentionner le 21 juillet en plus du chiffre déjà
   connu (99,67 % relatif à 32,8 % conservé).
3. **La correction combinée pour ViT (`mean_js` + λ=-1,3) est un résultat négatif net**, à présenter
   comme tel : bonne accuracy, mais perd la sélectivité attention qui faisait l'intérêt de
   `mean_js`. Le rapport précédent listait cette combinaison comme piste prometteuse — cette nuit
   la referme plutôt qu'elle ne la confirme.
4. `runs/vit_small_freeze_fix_lambda16/` manquant sur disque (§0) — à vérifier auprès de qui a pu
   le supprimer avant de considérer ce point clos.
