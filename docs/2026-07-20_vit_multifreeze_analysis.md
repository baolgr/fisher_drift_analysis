# Analyse des runs ViT multi-freeze (2026-07-20)

`runs/vit_small_freeze_fix_lambda13_multifreeze/` et
`runs/vit_small_freeze_fix_meanjs_multifreeze/` sont revenus du cluster (les 2 runs ResNet50
équivalents ne sont pas encore de retour). Rappel du design (`slurm/README.md` §"Multi-freeze
batch") : mêmes configs que les runs single-event `vit_small_freeze_fix_lambda13`/`_fix_meanjs`,
mais `freeze_interval: 7200 -> 4000`, donc jusqu'à 3 déclenchements du seuil de gel (steps 4000,
8000, 12000) au lieu d'un seul.

## 0. Bug de plotting trouvé et corrigé en préparant cette analyse

Le marqueur vertical "freeze" ajouté hier ne traçait qu'**un seul** repère (à
`cfg["freeze_interval"]`), donc ratait complètement les 2e/3e événements sur ces runs
multi-freeze — sur `fisher_drift.png` de `..._fix_meanjs_multifreeze`, l'effondrement de
l'accuracy au step 8000 n'avait aucun repère visuel avant correction. Corrigé : `plot_metric_evolution`
prend maintenant `freeze_steps` (liste), remplie en relisant les vraies lignes `[Freeze] Step N:`
de `train.log` (`src/utils/logs.py::parse_freeze_steps`) plutôt que de calculer des multiples de
`freeze_interval` depuis la config — important parce que l'early stopping peut couper un run avant
qu'un événement prévu n'ait lieu (voir `_fix_meanjs_multifreeze` ci-dessous, arrêté avant le 3e
événement). `scripts/regenerate_plots.py` mis à jour et relancé sur ces 2 runs pour appliquer le
correctif sans réentraîner. Suite de tests : 56 → 60 (nouveau `tests/test_logs.py` +
2 tests plotting), tout passe.

## 1. Résultats chiffrés

| Run | test accuracy | accuracy relative* | params entraînables | % conservé | epochs |
|---|---|---|---|---|---|
| `vit_small_nofreeze` (réf.) | 0,7273 | 100,0 % | 2 680 906 | 99,5 % | 40/40 |
| `vit_small_freeze_fix_lambda13` (single-event) | 0,7212 | 99,17 % | 1 796 170 | 66,7 % | 40/40 |
| **`..._fix_lambda13_multifreeze`** | **0,6921** | **95,16 %** | **503 914** | **18,7 %** | 40/40 |
| `vit_small_freeze_fix_meanjs` (single-event) | 0,7095 | 97,55 % | 1 316 362 | 48,9 % | 40/40 |
| **`..._fix_meanjs_multifreeze`** | **0,5074** | **69,76 %** | **23 770** | **0,9 %** | 31/40 (arrêt anticipé) |

*relative à `vit_small_nofreeze`.

## 2. `lambda13_multifreeze` : la progressivité coûte de l'accuracy, mais reste raisonnable

Les 3 événements de gel :

```
Step  4000: mean=0.4910 std=0.2248 λ=-1.300 threshold=0.1987 | frozen=22  remaining=386
Step  8000: mean=0.6104 std=0.2440 λ=-1.300 threshold=0.2932 | frozen=16  remaining=370
Step 12000: mean=0.6831 std=0.2457 λ=-1.300 threshold=0.3637 | frozen=84  remaining=286
```

Le seuil (`mean + λ·std`) **augmente** à chaque événement (0,199 → 0,293 → 0,364) — attendu,
`total_variation` étant une somme cumulée depuis le début de l'entraînement (voir
`docs/2026-07-19_vit_freeze_tuning_5_experiments_report.md` §0) : plus on avance dans
l'entraînement, plus la moyenne de la population (même réduite) grossit. Les deux premiers
événements gèlent peu (22, 16 chunks) ; le 3e en gèle 84 d'un coup — un rythme qui **accélère**
plutôt que de rester stable, cohérent avec un seuil qui grimpe plus vite que les scores des
chunks restants.

Choc de gel mesuré sur l'accuracy de validation, à chaque événement :

| Événement | epoch avant | epoch après | choc |
|---|---|---|---|
| step 4000 (~epoch 11) | 0,6059 | 0,6070 | +0,1 pt (négligeable) |
| step 8000 (~epoch 22) | 0,6643 | 0,6498 | -1,5 pts (mineur, récupéré en 1 epoch) |
| step 12000 (~epoch 34) | 0,6885 | 0,6217 | **-6,7 pts** (récupéré en 1-2 epochs) |

Le choc scale avec la **taille** de l'événement (22/16/84 chunks), pas juste avec le nombre
d'événements déjà passés — cohérent avec le mécanisme déjà documenté (§2 du rapport du 19 juillet)
transposé à un contexte progressif.

**Résultat le plus intéressant** : contrairement au run single-event à la même config (λ=-1,3,
`total_variation`), qui ne gèle **jamais** l'attention ni les Norm
(`docs/2026-07-19_vit_freeze_tuning_5_experiments_report.md` §3/§5 — 0 chunk Attn, 0 chunk Norm
gelés), la version progressive **atteint l'attention** : sur la population complète (violon
`chunk_drift_violin_by_param_type.png`), 76 chunks Attn gelés sur 192 (39,6 %) et 2 chunks Norm sur
104. Sur les 7 couches repères (`relative_update.png`), c'est concentré sur le 3e événement :
`blocks.2.attn.qkv`, `blocks.0.mlp.fc1` et `head` tombent à 0 exactement au step 12000, alors que
`blocks.0.attn.qkv` et `blocks.5.attn.qkv` continuent d'évoluer jusqu'à la fin. **`total_variation`
n'est donc pas structurellement aveugle à l'attention** — répéter le gel avec la même config finit
par l'atteindre, contrairement à ce que le sweep single-event du 19 juillet laissait penser. Une
piste concrète pour la réunion : la sélectivité par type de paramètre de `mean_js` (§3 du rapport
précédent) n'est peut-être pas une propriété unique de cette métrique, juste quelque chose que
`total_variation` met plus longtemps/plus d'événements à révéler.

## 3. `meanjs_multifreeze` : effondrement au 2e événement, pas au 3e

```
Step 4000: mean=0.3073 std=0.1362 λ=-1.000 threshold=0.1711 | frozen=54  remaining=354
Step 8000: mean=0.4513 std=0.1750 λ=-1.000 threshold=0.2763 | frozen=115 remaining=239
[EarlyStopping] Triggered — stopping training.  (epoch 31/40, 3e événement jamais atteint)
```

Le 2e événement fait chuter la population de 354 chunks/1 684 714 params à 239 chunks/**23 770
params** — la quasi-totalité de la masse de paramètres disparaît alors que seuls 115/354 = 32 %
des *chunks* sont gelés (même biais "gros tenseurs gelés en priorité" que d'habitude, mais
brutalement concentré sur un seul événement). Choc d'accuracy correspondant :

| Événement | epoch avant | epoch après | choc |
|---|---|---|---|
| step 4000 (~epoch 11) | 0,6059 | 0,5902 | -1,6 pt (mineur) |
| step 8000 (~epoch 22) | 0,6619 | **0,2203** | **-44,2 pts** |

Récupération très lente et incomplète avant l'arrêt anticipé (0,34 → 0,41 → 0,44 → ... → 0,51 sur
les 9 epochs suivantes) — quasi identique en ampleur et en forme au run single-event
`meanjs_lambda07` déjà catastrophique (`docs/2026-07-20_overnight_batch_analysis.md` §3, 23 386
params retenus, 0,5206 accuracy finale — la coïncidence numérique avec ce run-ci, 23 770 params,
0,5074, n'est probablement pas un hasard : les deux atterrissent dans le même régime
d'effondrement quasi-total).

**Écart avec l'extrapolation faite avant de lancer ce batch** (voir l'en-tête de
`config_vit_small_cluster_fix_meanjs_multifreeze.yaml`) : l'estimation projetait ~54 % de chunks
conservés après 3 événements en supposant une fraction gelée par événement à peu près stable. La
réalité est bien plus non-linéaire — le 2e événement à lui seul a été très supérieur au 1er (115
contre 54 chunks, mais surtout un saut brutal en *params*), invalidant l'hypothèse d'un taux
constant. À retenir pour tout futur calcul de risque sur ce type de config : extrapoler linéairement
depuis un seul run single-event sous-estime le risque de compounding pour `mean_js` spécifiquement
— `total_variation` (§2) s'est révélé nettement plus stable événement après événement.

Malgré l'effondrement, le violon confirme que la sélectivité attention de `mean_js` **persiste et
s'intensifie** sous gel progressif : 96 chunks Attn gelés sur 192 (50,0 %, contre 18,75 % en
single-event) et 48 chunks MLP-FC sur 96 (50,0 %, contre 29,2 %) — mean_js continue de cibler
l'attention même en régime agressif, simplement au prix d'une accuracy inacceptable à ce niveau
d'agressivité cumulée.

## 4. Bottom line

1. **Le bug du marqueur de gel unique est corrigé** (§0) — tout futur run multi-freeze sera
   maintenant tracé correctement dès la génération initiale, plus besoin de `regenerate_plots.py`
   après coup.
2. **`lambda13` en version progressive est le résultat le plus exploitable de ce batch** : coût
   d'accuracy réel mais modéré (95,16 % relatif, contre 99,17 % en single-event) pour un gel bien
   plus profond (81,3 % des params gelés contre 33,3 %) et, fait nouveau, une vraie sélectivité sur
   l'attention par bloc — un point à présenter comme réponse plus riche à la demande de
   l'encadrant (`docs/plan.md` point 7) que les runs single-event précédents.
3. **`meanjs` en version progressive confirme que son point de rupture (déjà vu en single-event à
   λ=-0,7) est atteignable aussi par répétition d'une config par ailleurs sûre** (λ=-1,0) — la
   prudence porte autant sur "combien de fois on gèle" que sur "à quel seuil". À ne pas présenter
   comme résultat exploitable en l'état (69,76 % relatif), mais utile comme mise en garde
   méthodologique.
4. En attente des 2 runs ResNet50 multi-freeze (`lambda13`/`meanjs`) pour savoir si ce même
   contraste (total_variation stable, mean_js fragile sous répétition) se reproduit sur l'autre
   architecture.
