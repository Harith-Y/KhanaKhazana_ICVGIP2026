# Tier class lists

`popular_{N}cls.json` is the sorted list of dish classes (folder names) of every tier reported in the paper.
`popular_1700cls_ranked_order.json` is the ranking of the 1,700-class tier (popularity score first, then image count);
the 400-, 500-, 1000-, 1300- and 1500-class tiers are its prefixes.

| Tier | Classes | Built from | Classes with no images in the final corpus |
|---|---|---|---|
| 100 | 100 | earlier corpus snapshot (13 Aug 2026 manifests) | 2 |
| 200 | 200 | earlier snapshot | 3 |
| 300 | 300 | earlier snapshot | 7 |
| 400 | 400 | final corpus (rebuilt 22-23 Oct 2026) | 0 |
| 500 | 500 | final corpus | 0 |
| 600 | 600 | earlier snapshot | 45 |
| 700 | 700 | earlier snapshot | 53 |
| 800 | 800 | earlier snapshot | 60 |
| 900 | 900 | earlier snapshot | 68 |
| 1000 | 1000 | final corpus | 0 |
| 1300 | 1300 | final corpus | 0 |
| 1500 | 1500 | final corpus | 0 |
| 1700 | 1700 | final corpus | 0 |

The 100-300 and 600-900 class tiers were trained on the earlier snapshot. Some of their classes were later merged or removed
during cleaning, so a tier rebuilt from the final corpus with `prepare_data_by_class_list.py` has fewer classes than the number in
the table. The lists here are the lists that were trained on. Tiers rebuilt on the final corpus (400, 500, 1000, 1300, 1500, 1700)
have every class present. The popularity ranking covers the first ~600 classes; later classes are ranked by image count.
