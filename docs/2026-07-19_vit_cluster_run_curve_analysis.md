# ViT-small — analyse des courbes `runs/vit_small_{freeze,nofreeze}` (2026-07-19)

Analyse des 16 PNG déjà générés dans `runs/vit_small_freeze/metrics_plots/` et
`runs/vit_small_nofreeze/metrics_plots/` (les 4 courbes curées + `all_metrics_grid` +
les 3 plots Appendix B.1 par run), lue à la lumière des critères de `CLAUDE.md` et
`docs/plan.md`. Aucun nouveau run, aucun nouveau script de plot — uniquement une lecture
des artefacts déjà produits par `train.py` (cf. `feedback_analyze_existing_artifacts_first`).

## 0. Ce run n'est pas le run local à 18 epochs déjà analysé

Premier constat, avant toute lecture de courbe : `runs/vit_small_freeze/summary.json` et
`runs/vit_small_nofreeze/summary.json` ont `num_epochs: 40` et `freeze_interval: 7200` —
ce sont les valeurs de **`config_vit_small_cluster.yaml`**, pas de `config_vit_small.yaml`
(18 epochs / `freeze_interval: 3200`). C'est donc un run différent de celui déjà documenté
dans `docs/2026-07-19_vit_v5_curve_analysis.md` (`runs/vit_small_freeze_v5_pure_knobs_lambda-1.0/`,
resté intact sur disque, non touché ici). `CLAUDE.md` indiquait la préparation cluster comme
faite mais **aucun run cluster encore exécuté** — ces deux dossiers sont donc un résultat
nouveau qui met à jour cet état, probablement la source du commit "nouveaux resultats".

Comparaison des deux résultats :

| | v5 local (18 epochs, `freeze_interval=3200`) | cluster config (40 epochs, `freeze_interval=7200`) |
|---|---|---|
| test_accuracy freeze | 0.6502 | 0.6574 |
| test_accuracy nofreeze | 0.6543 | 0.7273 |
| accuracy relative (freeze/nofreeze) | 99.4 % | **90.4 %** |
| params entraînables retenus | 48.9 % | **9.1 %** (245 818 / 2 693 578) |

Le rappel utilisateur `feedback_prefer_original_params` (préférer les knobs existants à du
code neuf) reste respecté ici — même config Fisher/gel que v5 (`js_variance_lambda=-1.0`,
`fisher_ema_interval=200`), seuls `num_epochs`/`freeze_interval` changent, comme prévu par
le fichier cluster. Mais le comportement de gel obtenu est nettement plus agressif qu'à
18 epochs : ce n'est pas un artefact de lecture, c'est visible et cohérent sur les 16 figures
(détail ci-dessous) — **à signaler avant utilisation en réunion**, voir §5.

## 1. Checks de correction du pipeline (critères `plan.md`/`CLAUDE.md`)

- **Métrique-vs-itération uniquement** : confirmé sur les 8 figures curées (x=step, y=une
  métrique, une ligne par couche) — jamais de scatter métrique-vs-métrique.
- **Aucun NaN / aucune courbe plate à zéro** sur les 7 couches curées, dans les deux runs.
- **`nofreeze` est un témoin propre** : `fisher_drift`, `fisher_magnitude`, `grad_norm`,
  `relative_update` de `vit_small_nofreeze` sont des courbes lisses, saturantes, sans
  discontinuité sur toute la durée (0 → 14 000 steps) — confirme que la mécanique de mesure
  elle-même ne produit aucun artefact en l'absence de gel, sur un horizon 2× plus long que
  la validation précédente à 18 epochs.
- **Couches d'attention actives** : `blocks.{0,2,5}.attn.qkv` suivent la même croissance saine
  que le reste dans `fisher_drift` (aucune plate à zéro) — reconfirme que le préfix-match
  `.attn.qkv` fonctionne aussi sur la config cluster.

## 2. Courbes curées (freeze) : un gel beaucoup plus large qu'à 18 epochs

Contrairement à v5 (où seules 2 des 7 couches suivies étaient franchement coupées, gel
partiel/étagé), ici **les 6 couches non-`patch_embed` s'effondrent simultanément** au même
step (~7200/7300, coïncide avec `freeze_interval=7200`) :

- `grad_norm` : `blocks.0/2/5.attn.qkv`, `blocks.0.mlp.fc1`, `blocks.5.mlp.fc2`, `head`
  tombent tous de leur bande ~0.1–1.2 à quasi-zéro juste après le step de gel. Seul
  `patch_embed.proj` continue dans sa bande bruitée ~1.5–2.7, inchangée avant/après.
- `fisher_magnitude` : les 6 mêmes couches convergent brutalement vers ~0.20 (plafond
  apparent) au même step ; `patch_embed.proj` reste seul à ~0.044, stable.
- `relative_update` : toutes les couches gelées retombent à un plateau quasi-nul après un pic
  transitoire à la transition (`blocks.5.mlp.fc2` culmine à ~0.33, le point le plus haut de
  toute la figure) — même signature de "freeze-shock" déjà documentée pour v5, mais ici sur
  6 couches à la fois plutôt que 2.

Cohérent avec `summary.json` : 9.1 % de params retenus, donc un gel qui touche la quasi-totalité
du réseau au-delà de `patch_embed.proj`, pas un gel étagé/partiel comme à 18 epochs.

## 3. Point à vérifier : le saut de `fisher_drift` *à la hausse* pile au step de gel

Dans `fisher_drift.png` (freeze), les couches qui viennent d'être gelées (`blocks.5.mlp.fc2`,
`head`, `blocks.5.attn.qkv`, etc.) sautent de ~0.51–0.55 à ~0.80–0.83 **exactement** au step de
gel, puis restent quasi plates ensuite — alors que `grad_norm`/`relative_update` montrent au
même instant que ces couches viennent de cesser d'apprendre. Une couche gelée qui affiche son
plus fort saut de dérive Fisher au moment précis où elle arrête de bouger est contre-intuitif.

Deux lectures possibles, non tranchées ici (même prudence épistémique que le "freeze-shock"
resté ouvert dans `docs/2026-07-19_vit_v5_curve_analysis.md`) :
- artefact de mesure : la courbe traitée ici est la dernière lecture EMA JS-distance
  (`trainer._chunk_js_history[k][-1][1]`, cf. la mise en garde déjà notée dans `CLAUDE.md`/
  l'analyse v5 — pas littéralement le score de gel `total_variation`) ; si l'EMA cesse d'être
  mise à jour pour un chunk gelé pendant qu'elle continue d'évoluer pour la distribution de
  référence, un saut pourrait apparaître mécaniquement sans changement réel de poids.
- signal réel mais non lié au poids gelé lui-même : une activation en aval qui change
  (couches encore entraînables) peut modifier la statistique Fisher collectée en amont d'une
  couche gelée pendant le forward, même si son propre gradient est masqué à zéro après coup.

Les plots par population complète (heatmap/violon, §4) montrent que ce plafond ~0.83 est
atteint aussi bien en `nofreeze` (sans aucun gel) qu'en `freeze` — donc le plafond en lui-même
n'est pas un artefact du gel. Ce qui reste à vérifier séparément est seulement la *simultanéité
exacte* du saut avec l'instant de gel pour les couches concernées. À creuser dans
`src/fisher/trainer.py`/`fisher_core.py` si ces courbes sont montrées telles quelles à
l'encadrant — sinon, préciser la mise en garde déjà écrite dans l'analyse v5 (§ "précision
caveat").

## 4. Plots population complète (heatmap / tiers / violon) — cohérence freeze vs nofreeze

- **Plafond de dérive ~0.83 pour la majorité des chunks, dans les deux runs** : le heatmap et
  le plot par tiers de `nofreeze` (sans gel, 14 000 steps) montrent exactement la même forme
  que `freeze` — un tiers "high-drift" (n=6) qui sature vers ~0.83 dès ~8000–10000 steps, un
  tiers "mid-drift" (n=4) qui plafonne ~0.30, un tiers "low-drift" (n=6) ~0.15–0.30. C'est un
  bon signe de robustesse : le plafond n'est pas causé par le gel (il apparaît aussi sans gel),
  probablement une propriété du mode `js_distance_mode="log"` sur un entraînement de cette
  longueur plutôt qu'un bug introduit par le mécanisme de gel.
- **Différence de taille de population entre les deux violons** : `freeze` a nettement moins
  de chunks trackés dans les catégories Attn (n=120 vs 192 en nofreeze) et MLP-FC (n=48 vs 96)
  qu'en nofreeze, alors que Norm (101 vs 104) et Other/Conv-FC (12 vs 16) sont presque
  inchangés. C'est cohérent avec §2 : le gel touche presque exclusivement attention et MLP,
  épargnant patch_embed/norm — et confirme au niveau de la population complète de chunks
  (pas seulement les 7 couches curées) la même asymétrie déjà vue sur les courbes curées.
- Le violon `MLP-FC` de `freeze` est quasi une aiguille plate à 0.83 (zéro spread visible) —
  cohérent avec "tous les chunks MLP restants ont saturé au même plafond avant le gel", pas
  un défaut de rendu.

## 5. Bottom line

Le pipeline de mesure lui-même passe les checks de `plan.md`/`CLAUDE.md` : format
métrique-vs-itération respecté partout, attention non-nulle, `nofreeze` propre et cohérent sur
tout l'horizon 40 epochs, les 4 métriques + les plots de population s'accordent entre eux sur
le même découpage de couches. Le doute du §3 (saut de dérive au moment du gel) est à vérifier
mais n'invalide pas la lecture qualitative globale.

Ce qui mérite une décision avant le 21 juillet : **ce run gèle beaucoup plus large et coûte
plus cher en accuracy que la config v5 déjà validée** (9.1 % de params retenus et 90.4 %
d'accuracy relative, contre 48.9 % et 99.4 %). La config cluster n'a changé que `num_epochs`
et `freeze_interval` (mis à l'échelle comme documenté dans le fichier de config lui-même) —
rien d'anormal dans la configuration en soi, mais le résultat qualitatif du gel change
nettement à 40 epochs. À trancher avec l'utilisateur : présenter ce résultat tel quel (le gel
plus agressif reste un résultat valide, juste différent), ou retravailler
`freeze_interval`/`js_variance_lambda` pour l'horizon 40 epochs comme cela avait été fait pour
la config 18 epochs (cf. `fisheradaptune_freeze_collapse_bug` en mémoire). Pour rappel,
l'objectif du jalon du 21 est de valider le pipeline, pas l'accuracy — mais un écart aussi
large entre config locale et cluster vaut la peine d'être noté explicitement dans la
présentation, plutôt que de laisser croire que 48.9 %/99.4 % est *le* résultat FisherAdapTune
sur ce benchmark.

Note annexe, non traitée ici : par convention (`CLAUDE.md`), `runs/vit_small_freeze/` et
`runs/vit_small_nofreeze/` sont les dossiers de sortie par défaut (non versionnés) — à
renommer vers un nom explicite (style `runs/vit_small_freeze_cluster40ep_v1/`) si ce résultat
doit être conservé, avant qu'un futur run à ces mêmes chemins ne l'écrase.

## 6. Addendum (récupération du vrai log cluster) — le §3/§5 sont expliqués, pas juste constatés

Le stdout réel du job SLURM (`slurm/logs/vit_freeze-16724978.out`, récupéré après coup —
`train.py` ne sauvegardait pas encore `train.log` au moment de ce run) donne la ligne
`_apply_variance_freeze` qui manquait :

```
[Freeze] Step 7200: variation mean=0.5744 std=0.2527 λ=-1.000 threshold=0.3217 | frozen=127 remaining=281
[Freeze] Step 7200: trainable tensors=59 chunks=281 params=245,818
```

**Le chiffre clé, invisible dans les PNG** : au niveau des *chunks*, seuls 127/408 (31 %) sont
gelés, 281/408 (69 %) survivent — un gel qui semble presque équilibré. Mais au niveau des
*paramètres*, les 245 818 params qui survivent (9,2 % de 2 680 906) donnent une taille moyenne
de ~875 éléments par chunk survivant, contre ~19 175 éléments par chunk gelé — un **facteur
~22×**. Le gel n'a donc pas coupé "un peu partout" ; il a systématiquement gelé les *gros*
chunks (blocs de lignes de grandes matrices de poids — attn.qkv/mlp.fc1/fc2, cohérent avec §2)
et épargné les *petits* (biais, affines LayerNorm, ~192 éléments ou moins) — exactement le
biais déjà documenté en mémoire (`fisheradaptune_freeze_collapse_bug` : les petits tenseurs
produisent des histogrammes JS-distance à 64 bins beaucoup plus bruités sur peu d'éléments,
donc un score de dérive artificiellement gonflé par le bruit d'échantillonnage plutôt que par
un vrai signal). Ce mémo documentait ce biais pour l'ancienne config buggée
(`lambda=1.0`) où il causait l'effondrement inverse (les petits tenseurs bruités passaient
*au-dessus* du seuil et survivaient seuls) ; ici, avec `lambda=-1.0` (seuil bas, censé garder
la quasi-totalité), le même biais gonfle `mean`/`std` via les petits chunks bruités, ce qui
repousse par ricochet les gros chunks — dont le score `total_variation` réel a en grande partie
cessé de croître une fois saturé près du plafond ~0.83 vu au heatmap (§4) — sous le seuil
`mean - std`. C'est une lecture cohérente, mais reste une hypothèse mécanistique, pas une preuve
directe (pas de dump par-chunk des scores individuels dans ce log) — même prudence que pour le
reste du document.

Ça répond directement à la question ouverte du §3/§5 ("pourquoi 91 % de params coupés au lieu
de ~51 %") : ce n'est pas un dérèglement aléatoire, c'est le même biais petit-tenseur déjà connu,
mais amplifié par l'horizon plus long (36 mises à jour EMA avant la coupe, contre ~16 à
18 epochs) qui laisse plus de temps au bruit cumulatif des petits chunks de gonfler `total_variation`
(une somme de deltas absolus n'annule jamais le bruit, elle l'accumule) pendant que les gros
chunks, eux, plafonnent en `total_variation` réel une fois leur dérive saturée.

**Conséquence pour la discussion lambda/interval déjà eue** : ça renforce nettement l'idée
"passer `chunk_selection_metric` à `mean_js`" déjà évoquée comme piste secondaire — `mean_js`
est une *moyenne* de lectures JS (`js_sum/js_count`), qui ne grossit pas avec le nombre de
mises à jour EMA observées, contrairement à `total_variation` qui est une somme cumulée. Ça
n'élimine pas le bruit d'échantillonnage propre aux petits tenseurs (biais séparé, documenté
lui aussi en mémoire), mais ça retire le second facteur — la durée d'accumulation — qui est
précisément ce qui différencie la config 18 epochs (marchait) de la config 40 epochs (collapse).
Un sweep lambda seul, sans changer la métrique, risque de rester sensible au même effet à mesure
que l'horizon s'allonge encore (ex. si un futur run passe à plus de 40 epochs).

**Confirmation indépendante au niveau de la validation** : le log epoch-par-epoch montre le
"freeze-shock" déjà nommé dans `docs/fisheradaptune_freeze_experiments.md`, ici visible
directement sur la courbe de val-accuracy agrégée (pas seulement par couche comme dans l'analyse
v5) : epoch 20 (juste avant la coupe) `accuracy=0.6678`, epoch 21 (juste après)
`accuracy=0.4951` (-17,3 points, `loss` 0.97→1.39), récupération progressive sur ~7 epochs sans
jamais redépasser franchement le pic pré-gel (meilleur post-gel : epoch 28, `accuracy=0.6617`,
encore sous 0.6678), puis `EarlyStopping` déclenché à l'epoch 32. Le `test_accuracy=0.6574`
final est cohérent avec ce plateau post-choc, pas avec le pic pré-gel — une partie du coût
d'accuracy vs. `nofreeze` (0.7273) vient donc autant de ce plateau de récupération que du volume
de params gelés en lui-même.
