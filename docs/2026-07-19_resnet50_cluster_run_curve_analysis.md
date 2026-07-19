# ResNet50 — analyse des courbes `runs/resnet50_{freeze,nofreeze}` (2026-07-19)

Analyse des artefacts déjà produits par `train.py` dans `runs/resnet50_freeze/` et
`runs/resnet50_nofreeze/` (`summary.json`, `train.log`/`.out` SLURM, les 4 courbes curées +
`all_metrics_grid` + les 3 plots Appendix B.1 par run), lue à la lumière des critères de
`CLAUDE.md` et `docs/plan.md`. Aucun run relancé, aucun fichier modifié — uniquement une lecture
des artefacts existants (cf. mémoire `feedback_analyze_existing_artifacts_first`). Miroir de
`docs/2026-07-19_vit_cluster_run_curve_analysis.md`, même méthode, pour permettre une
comparaison directe ResNet-vs-ViT au même point de config.

## 0. Ce run n'est pas documenté dans l'état courant du projet — à signaler

Premier constat, avant toute lecture de courbe : `summary.json` des deux dossiers a
`num_epochs: 40` et `freeze_interval: 7200` — ce sont les valeurs de
**`config_resnet50_cluster.yaml`**, pas de `config_resnet50.yaml` (18 epochs). Les `.out` SLURM
(`resnet_freeze-16724992.out`, `resnet_nofreeze-16724986.out`, modules Compute Canada,
`/localscratch/blgr.*`) confirment un run cluster réel.

Or :
- `CLAUDE.md` (« Current status ») dit explicitement : *« ResNet50 config prepared … but runs
  not yet executed — deliberately not run this session »*.
- L'en-tête de `src/configs/config_resnet50_cluster.yaml` dit littéralement : *« Still NOT YET
  RUN for ResNet50 at any epoch count »*.

Les deux affirmations sont maintenant **fausses** — ces deux runs cluster existent bel et bien
sur disque (mêmes horodatages que le commit "new jobs" / "nouveaux resultats" du 19 juillet).
Comme pour `vit_small_freeze`/`vit_small_nofreeze` (déjà signalé dans l'analyse ViT équivalente),
ces dossiers écrivent dans le chemin par défaut non versionné `runs/<model>_<freeze|nofreeze>/` —
à renommer vers un nom explicite (style `runs/resnet50_freeze_cluster40ep_v1/`) avant qu'un futur
run n'écrase ces résultats, et **`CLAUDE.md`/l'en-tête du config sont à mettre à jour** pour
refléter que ce run a eu lieu (je ne les modifie pas ici, cf. consigne de cette tâche).

## 1. Checks de correction du pipeline (critères `plan.md`/`CLAUDE.md`)

- **Métrique-vs-itération uniquement**, jamais métrique-vs-métrique : confirmé sur les 4 figures
  curées et `all_metrics_grid` (x = step, y = une métrique, une ligne par couche).
- **`nofreeze` est un témoin propre** : `fisher_drift`, `fisher_magnitude`, `grad_norm`,
  `relative_update` de `resnet50_nofreeze` sont des courbes lisses, sans discontinuité, sur tout
  l'horizon 0→11 200 steps — la mécanique de mesure elle-même ne produit aucun artefact en
  l'absence de gel.
- **Les deux runs sont identiques jusqu'au gel**, comme attendu (même seed, même split, même
  schedule) : `train.log` montre des chiffres `train_loss`/`accuracy` **bit-à-bit identiques**
  epoch 1 à 20 entre `resnet50_freeze` et `resnet50_nofreeze` (ex. epoch 10 :
  `accuracy=0.8486` dans les deux) — confirme qu'aucune divergence (bug d'implémentation,
  non-déterminisme) n'existe avant l'instant du gel lui-même.
- **Pas de couche "attention"** : conforme à la note de `plan.md` — ResNet50 n'a pas de bloc
  d'attention par construction ; le registre curé (7 couches) porte donc uniquement l'axe
  précoce/tardif et conv-vs-Linear-final, comme prévu.
- **`classify_param_type` reste exhaustif pour ResNet** : le violon par type de paramètre ne
  montre que deux catégories (`Norm`, `Other (Conv/FC)`) — cohérent, pas de crash de sidebar
  (cf. le bug `cls_token`/`pos_embed` déjà corrigé pour le ViT, sans équivalent ici puisque
  ResNet n'a aucun `nn.Parameter` brut hors module).

## 2. Résultat headline : accuracy comparable, deux tiers des paramètres gelés

| | `resnet50_nofreeze` | `resnet50_freeze` |
|---|---|---|
| test_accuracy | **0.9027** | **0.8997** |
| test_loss | 0.3719 | 0.3991 |
| params entraînables | 23 520 842 (100 %) | 7 717 578 (**32,8 %**) |
| accuracy relative (freeze/nofreeze) | — | **99,67 %** |

C'est le résultat que le point 4 de `plan.md` demande de produire pour le 21 juillet : à
**67,2 % des paramètres gelés**, la perte d'accuracy est de seulement **0,30 point** (0,90 % en
relatif). C'est un résultat nettement plus favorable au gel que celui obtenu sur ViT à la même
config cluster (voir §6) — un point positif à mettre en avant.

## 3. Courbes curées (freeze) : quelles couches sont gelées, lesquelles survivent

Le gel se déclenche une seule fois, au step 7200 (= début de l'epoch 21, `351` steps/epoch),
conforme à `freeze_interval=7200` :

```
[Freeze] Step 7200: variation mean=0.5611 std=0.2529 λ=-1.000 threshold=0.3082 | frozen=152 remaining=492
[Freeze] Step 7200: trainable tensors=124 chunks=492 params=7,717,578
```

Sur les 7 couches curées, le partage est net :
- **Gelées** (JS-drift, magnitude Fisher et grad-norm retombent exactement à 0 juste après le
  step 7200) : `layer1.0.conv2`, `layer2.2.conv2`, `layer3.3.conv2`, `layer4.2.conv2` — les
  quatre convolutions 3×3 « profondes » choisies comme représentatives de chaque étage
  résiduel (cf. `plan.md` §layers.py, choisies précisément parce que ce sont les vraies convs
  spatiales, pas des projections 1×1).
- **Non gelées**, continuent d'évoluer normalement après 7200 : `stem_conv`, `layer4.2.conv3`
  (la 1×1 d'expansion de canaux) et `fc`.

Cette asymétrie **grosses convs 3×3 gelées / petites couches (stem, 1×1, fc) épargnées** est
quantifiable au niveau global (pas seulement les 7 couches curées) : 152 chunks gelés
totalisent 15 803 264 paramètres (≈103 995 éléments/chunk en moyenne), contre 492 chunks
retenus totalisant 7 717 578 paramètres (≈15 686 éléments/chunk en moyenne) — un facteur
**~6,6×**. C'est le même biais « les gros tenseurs sont jugés plus stables et gelés
préférentiellement » déjà documenté pour le ViT (`docs/2026-07-19_vit_cluster_run_curve_analysis.md`
§6, facteur ~22× là-bas, et mémoire `fisheradaptune_freeze_collapse_bug`) — présent ici aussi,
en plus modéré. Cohérent avec le fait que `chunk_selection_metric: total_variation` est une somme
cumulée de deltas JS absolus : sur un tenseur à peu d'éléments (biais BatchNorm, `fc` à 20 480
params, `stem_conv` à 1 728 params), l'estimation JS-distance par histogramme à 64 bins est plus
bruitée, gonflant artificiellement `total_variation` et faisant passer ces petits tenseurs
au-dessus du seuil de gel — pas nécessairement parce qu'ils dérivent réellement plus.

## 4. Choc de gel (« freeze-shock ») — mesurable directement sur l'accuracy de validation

Le même phénomène que documenté pour le ViT (`docs/fisheradaptune_freeze_experiments.md`,
`docs/2026-07-19_vit_cluster_run_curve_analysis.md` §Addendum) est visible ici, en plus modéré :

| epoch | `freeze` val accuracy | `nofreeze` val accuracy |
|---|---|---|
| 20 (juste avant le gel) | 0.8936 | 0.8936 |
| 21 (juste après le gel) | **0.7578** (Δ = −0,1358) | 0.8701 (Δ = −0,0235) |
| 25 (retour au niveau pré-gel) | 0.8943 | 0.8898 |
| 26 (premier epoch qui dépasse le pic pré-gel) | 0.8953 | 0.8961 |

Le choc « propre au gel » (au-delà du bruit epoch-à-epoch normal, visible aussi côté `nofreeze`
à −0,0235) est donc d'environ **−11,2 points** d'accuracy sur un seul epoch, suivi d'une
récupération complète en ~4 epochs. C'est cohérent avec `relative_update.png` : un pic massif
(jusqu'à 0,68 pour `layer4.2.conv3`, 0,57 pour `layer4.2.conv2` juste avant que ce dernier soit
mis à zéro) exactement au step de transition — l'optimiseur réagit brutalement à la
redistribution soudaine du signal de gradient sur un sous-ensemble de couches. Nettement moins
sévère que le choc ViT (−17,3 points, pas de retour franc au pic pré-gel avant l'arrêt anticipé,
cf. le même document §Addendum) — cohérent avec un gel moins agressif ici (67 % vs 91 % des
paramètres côté ViT).

Point notable et contre-intuitif, même phénomène déjà vu et laissé ouvert côté ViT (§3 de
l'analyse ViT) : dans `fisher_drift.png`, la courbe `fc` (non gelée) **saute vers le haut**
(≈0,49 → ≈0,80) exactement au step de gel, alors que `grad_norm`/`relative_update` montrent au
même instant une simple continuation de tendance pour cette couche, sans rupture équivalente.
Cohérent avec l'hypothèse déjà formulée côté ViT : un artefact possible de la lecture EMA de
`_chunk_js_history[k][-1]` au moment où la distribution de référence change brutalement pour de
nombreux chunks voisins gelés, plutôt qu'un vrai saut de dérive du poids `fc` lui-même. Pas
tranché ici non plus — signalé pour cohérence avec l'analyse ViT, pas une nouvelle piste.

## 5. Plots population complète (heatmap / tiers / violon)

- **Heatmap** (`chunk_drift_heatmap.png`) : les lignes à drift final faible (haut, bleu clair)
  sont majoritairement `Layer3-4 (late)` (orange), celles à drift élevé (bas, bleu foncé) sont un
  mélange `Stem`/`Layer1-2 (early)`/`Layer3-4` — pas de séparation stricte précoce/tardif comme
  chez le ViT, mais une tendance : les couches précoces et le stem dominent la queue haute.
- **Trajectoires par tier** (`chunk_drift_trajectories_by_tier.png`) : structure à 3 paliers nette
  et quasi identique en forme dans `freeze` et `nofreeze` jusqu'au step 7200 (tier haut plafonne
  ~0,83, tier moyen ~0,66, tier bas redescend après un pic ~0,30) — même plafond de dérive que
  celui documenté côté ViT, apparaissant aussi bien avec gel que sans, donc pas un artefact du
  mécanisme de gel lui-même (bon signe de robustesse de la métrique). Le tier bas de `freeze`
  montre un décrochage additionnel après ~8000 steps qui n'existe pas dans `nofreeze` :
  cohérent avec le gel qui vide une partie de la population plutôt qu'avec un bug.
- **Violon par type de paramètre** (`chunk_drift_violin_by_param_type.png`) : **`Norm` (BatchNorm)
  dérive nettement plus que `Other (Conv/FC)`** dans les deux runs — médiane 0,74 vs 0,34
  (`freeze`), 0,70 vs 0,24 (`nofreeze`). C'est la même signature « petits tenseurs = JS-distance
  gonflée par le bruit d'échantillonnage » que celle identifiée pour le ViT sur les affines
  LayerNorm — les paramètres BatchNorm (poids/biais par canal, quelques dizaines à quelques
  milliers d'éléments) sont structurellement plus petits que les tenseurs de conv qu'ils
  accompagnent. C'est un **second exemple indépendant (architecture différente) du même biais
  méthodologique**, ce qui renforce l'hypothèse déjà notée en mémoire plutôt que d'ouvrir un
  nouveau doute — utile à mentionner tel quel à l'encadrant comme limite connue de
  `chunk_selection_metric="total_variation"`, pas comme un bug de ce run précis.

## 6. Comparaison ResNet50 vs ViT-small (même config cluster, freeze_interval=7200)

| | ResNet50 | ViT-small |
|---|---|---|
| accuracy relative (freeze/nofreeze) | **99,67 %** | 90,4 % |
| params entraînables retenus | 32,8 % | 9,1 % |
| chocs de gel (Δ accuracy epoch du gel) | −11,2 pts (récupéré en 4 epochs) | −17,3 pts (jamais franchement récupéré avant arrêt anticipé) |
| facteur taille gros-chunks-gelés / petits-chunks-retenus | ~6,6× | ~22× |

Le gel guidé par Fisher est donc **nettement plus favorable sur ResNet50 que sur ViT** à cette
config : moins de paramètres gelés en proportion, un choc d'accuracy plus modéré et récupéré
complètement, un biais petit-tenseur/gros-tenseur moins extrême. Une lecture plausible (non
vérifiée directement ici) : les BatchNorm de ResNet sont beaucoup plus nombreux et répartis dans
tout le réseau que les LayerNorm du ViT relatif à sa taille totale, ce qui dilue davantage le
biais « petits tenseurs bruités » sur l'ensemble de la population de chunks. À creuser seulement
si l'encadrant pose la question — pour le jalon du 21, le point actionnable est que **les deux
architectures montrent le même biais qualitatif** (gros tenseurs gelés préférentiellement,
BatchNorm/LayerNorm survivent par bruit de mesure plutôt que par vraie stabilité), avec une
intensité différente selon l'architecture.

## 7. Bottom line pour la réunion du 21 juillet

- **Le pipeline de mesure passe les checks de `plan.md`/`CLAUDE.md`** : format
  métrique-vs-itération respecté partout, `nofreeze` propre sur tout l'horizon, les deux runs
  identiques jusqu'au gel (confirme l'absence de bug de non-déterminisme), cohérence entre
  courbes curées et plots de population complète.
- **Chiffre à retenir pour la comparaison avec/sans gel demandée au point 3 de `plan.md`** :
  ResNet50 conserve 99,67 % de l'accuracy avec 67,2 % des paramètres gelés — le résultat le plus
  favorable au gel des deux architectures benchmarkées à ce jour.
- **Deux points à signaler explicitement à l'encadrant, pas à corriger avant la réunion** :
  (a) `CLAUDE.md` et l'en-tête de `config_resnet50_cluster.yaml` affirment tous deux que ce run
  n'a pas eu lieu — à mettre à jour après la réunion, pas avant (hors scope de cette tâche
  d'analyse) ; (b) le biais « gros tenseur gelé préférentiellement / petit tenseur retenu par
  bruit de mesure » est confirmé sur une deuxième architecture indépendante — limite connue de
  `total_variation` comme `chunk_selection_metric`, pas un bug propre à ce run.
- Comme pour le run ViT équivalent : `runs/resnet50_freeze/` et `runs/resnet50_nofreeze/` sont
  les chemins de sortie par défaut non versionnés — à renommer avant qu'un futur run ne les
  écrase, si ce résultat doit être conservé pour la réunion.
