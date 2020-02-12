Association Rules Mining, Apriori Implimentation

Timothy Asp, Caleb Carlton

Input Format: python apriori.py [--no-rules] <dataFile-out1.csv> <minSup> <minConf>
   --no-rules will run the code without rules generation.
   The input datafile must be in the sparse vector format (see *-out1.csv in the different folders of ./data)

Example:  

   python apriori.py --no-rules data/1000/1000-out1.csv .03 .7
   python apriori.py data/example/out1.csv .03 .7

== Extended Bakery Printouts

=================================================================
Dataset:  data/example/out1.csv  MinSup:  0.03  MinConf:  0.7
==================================================================
   1 :   Blackberry Tart (15),   Apple Danish (36)    support= 0.139
   2 :   Gongolais Cookie (22),  Napoleon Cake (9)    support= 0.181
   3 :   Lemon Cake (1),   Single Espresso (49)    support= 0.127
   4 :   Apple Tart (12),  Berry Tart (14),  Blueberry Tart (16)  support= 0.257

Skyline Itemsets:  4

   Rule 1 :   Blackberry Tart (15)  --> Apple Danish (36)    [sup= 0.139  conf= 0.751351351351 ]
   Rule 2 :   Apple Danish (36)  --> Blackberry Tart (15)    [sup= 0.139  conf= 0.798850574713 ]
   Rule 3 :   Gongolais Cookie (22)    --> Napoleon Cake (9)    [sup= 0.181  conf= 0.841860465116 ]
   Rule 4 :   Napoleon Cake (9)  --> Gongolais Cookie (22)   [sup= 0.181  conf= 0.804444444444 ]
   Rule 5 :   Lemon Cake (1)  --> Single Espresso (49)    [sup= 0.127  conf= 0.814102564103 ]
   Rule 6 :   Single Espresso (49)  --> Lemon Cake (1)    [sup= 0.127  conf= 0.783950617284 ]
   Rule 7 :   Apple Tart (12),   Berry Tart (14)   --> Blueberry Tart (16) [sup= 0.257  conf= 0.958955223881 ]
   Rule 8 :   Apple Tart (12),   Blueberry Tart (16)  --> Berry Tart (14)   [sup= 0.257  conf= 0.992277992278 ]
   Rule 9 :   Berry Tart (14),   Blueberry Tart (16)  --> Apple Tart (12)   [sup= 0.257  conf= 0.996124031008 ]

=================================================================
Dataset:  data/75000/75000-out1.csv  MinSup:  0.033  MinConf:  0.7
==================================================================

   1:   Berry Tart (14),  Bottled Water (44)   support= 0.0378
   2:   Apricot Croissant (32), Hot Coffee (45)   support= 0.0353733333333
   3:   Strawberry Cake (4), Napoleon Cake (9)    support= 0.0431466666667
   4:   Casino Cake (2),  Chocolate Coffee (46)   support= 0.03524
   5:   Chocolate Tart (17), Vanilla Frappuccino (47)   support= 0.03596
   6:   Marzipan Cookie (27),   Tuile Cookie (28)    support= 0.05092
   7:   Blackberry Tart (15),   Coffee Eclair (7)    support= 0.0364133333333
   8:   Blueberry Tart (16), Hot Coffee (45)   support= 0.03504
   9:   Gongolais Cookie (22),  Truffle Cake (5)  support= 0.04392
   10:   Cheese Croissant (33), Orange Juice (42)    support= 0.0430666666667
   11:   Blueberry Tart (16),   Apricot Croissant (32)  support= 0.0435066666667
   12:   Lemon Cake (1),  Lemon Tart (19)   support= 0.0368533333333
   13:   Chocolate Cake (0), Chocolate Coffee (46)   support= 0.04404
   14:   Cherry Tart (18),   Opera Cake (3),   Apricot Danish (35)  support= 0.0411066666667
   15:   Chocolate Coffee (46), Chocolate Cake (0),  Casino Cake (2)   support= 0.0333866666667
   16:   Apple Pie (11),  Almond Twist (37),   Coffee Eclair (7)    support= 0.03432

Skyline Itemsets:  16

   Rule 1:   Cherry Tart (18),  Opera Cake (3)    --> Apricot Danish (35) [sup= 0.0411066666667  conf= 0.947740547187 ]
   Rule 2:   Cherry Tart (18),  Apricot Danish (35)  --> Opera Cake (3)    [sup= 0.0411066666667  conf= 0.77423405324 ]
   Rule 3:   Opera Cake (3), Apricot Danish (35)  --> Cherry Tart (18)     [sup= 0.0411066666667  conf= 0.955376510691 ]
   Rule 4:   Chocolate Cake (0),   Casino Cake (2)   --> Chocolate Coffee (46) [sup= 0.0333866666667  conf= 0.939587242026 ]
   Rule 5:   Apple Pie (11), Almond Twist (37)    --> Coffee Eclair (7)    [sup= 0.03432  conf= 0.935659760087 ]
   Rule 6:   Apple Pie (11), Coffee Eclair (7)    --> Almond Twist (37)    [sup= 0.03432  conf= 0.920930232558 ]
   Rule 7:   Almond Twist (37), Coffee Eclair (7)    --> Apple Pie (11)    [sup= 0.03432  conf= 0.924568965517 ]

=================================================================
Dataset:  data/20000/20000-out1.csv  MinSup:  0.03  MinConf:  0.7
==================================================================

   1:   Berry Tart (14),  Bottled Water (44)   support= 0.0357
   2:   Chocolate Tart (17), Walnut Cookie (29)   support= 0.03055
   3:   Strawberry Cake (4), Napoleon Cake (9)    support= 0.04455
   4:   Casino Cake (2),  Chocolate Coffee (46)   support= 0.0357
   5:   Almond Twist (37),   Hot Coffee (45)   support= 0.03085
   6:   Hot Coffee (45),  Coffee Eclair (7)    support= 0.0317
   7:   Blackberry Tart (15),   Single Espresso (49)    support= 0.03015
   8:   Chocolate Tart (17), Vanilla Frappuccino (47)   support= 0.03675
   9:   Marzipan Cookie (27),   Tuile Cookie (28)    support= 0.04855
   10:   Blackberry Tart (15),  Coffee Eclair (7)    support= 0.03675
   11:   Blueberry Tart (16),   Hot Coffee (45)   support= 0.0357
   12:   Gongolais Cookie (22), Truffle Cake (5)  support= 0.04335
   13:   Lemon Cake (1),  Lemon Tart (19)   support= 0.037
   14:   Cheese Croissant (33), Orange Juice (42)    support= 0.0439
   15:   Apple Pie (11),  Hot Coffee (45)   support= 0.03085
   16:   Blueberry Tart (16),   Apricot Croissant (32)  support= 0.04185
   17:   Chocolate Cake (0), Chocolate Coffee (46)   support= 0.04405
   18:   Walnut Cookie (29), Vanilla Frappuccino (47)   support= 0.03095
   19:   Chocolate Coffee (46), Chocolate Cake (0),  Casino Cake (2)   support= 0.0339
   20:   Apricot Croissant (32),   Hot Coffee (45),  Blueberry Tart (16) support= 0.0326
   21:   Apple Pie (11),  Almond Twist (37),   Coffee Eclair (7)    support= 0.03415
   22:   Cherry Tart (18),   Opera Cake (3),   Apricot Danish (35)  support= 0.041

Skyline Itemsets:  22

   Rule 1:   Chocolate Cake (0),   Casino Cake (2)   --> Chocolate Coffee (46) [sup= 0.0339  conf= 0.945606694561 ]
   Rule 2:   Apricot Croissant (32),  Hot Coffee (45)   --> Blueberry Tart (16) [sup= 0.0326  conf= 0.928774928775 ]
   Rule 3:   Apple Pie (11), Almond Twist (37)    --> Coffee Eclair (7)    [sup= 0.03415  conf= 0.949930458971 ]
   Rule 4:   Apple Pie (11), Coffee Eclair (7)    --> Almond Twist (37)    [sup= 0.03415  conf= 0.91677852349 ]
   Rule 5:   Almond Twist (37), Coffee Eclair (7)    --> Apple Pie (11)    [sup= 0.03415  conf= 0.942068965517 ]
   Rule 6:   Cherry Tart (18),  Opera Cake (3)    --> Apricot Danish (35) [sup= 0.041  conf= 0.939289805269 ]
   Rule 7:   Cherry Tart (18),  Apricot Danish (35)  --> Opera Cake (3)    [sup= 0.041  conf= 0.780209324453 ]
   Rule 8:   Opera Cake (3), Apricot Danish (35)  --> Cherry Tart (18)     [sup= 0.041  conf= 0.945790080738 ]

==================================================================
Dataset:  data/5000/5000-out1.csv  MinSup:  0.03  MinConf:  0.7
==================================================================
   1:   Strawberry Cake (4), Napoleon Cake (9)    support= 0.0422
   2:   Casino Cake (2),  Chocolate Coffee (46)   support= 0.0346
   3:   Almond Twist (37),   Hot Coffee (45)   support= 0.0336
   4:   Blackberry Tart (15),   Single Espresso (49)    support= 0.0314
   5:   Chocolate Tart (17), Vanilla Frappuccino (47)   support= 0.0348
   6:   Marzipan Cookie (27),   Tuile Cookie (28)    support= 0.0496
   7:   Apple Croissant (31),   Apple Danish (36)    support= 0.033
   8:   Apple Tart (12),  Apple Danish (36)    support= 0.0324
   9:   Blackberry Tart (15),   Coffee Eclair (7)    support= 0.0356
   10:   Blueberry Tart (16),   Hot Coffee (45)   support= 0.035
   11:   Gongolais Cookie (22), Truffle Cake (5)  support= 0.0472
   12:   Lemon Cake (1),  Lemon Tart (19)   support= 0.0336
   13:   Cheese Croissant (33), Orange Juice (42)    support= 0.043
   14:   Berry Tart (14), Bottled Water (44)   support= 0.0366
   15:   Blueberry Tart (16),   Apricot Croissant (32)  support= 0.044
   16:   Chocolate Cake (0), Chocolate Coffee (46)   support= 0.0394
   17:   Apple Tart (12), Apple Croissant (31)    support= 0.0316
   18:   Chocolate Coffee (46), Chocolate Cake (0),  Casino Cake (2)   support= 0.0312
   19:   Apricot Croissant (32),   Hot Coffee (45),  Blueberry Tart (16) support= 0.0328
   20:   Cherry Tart (18),   Opera Cake (3),   Apricot Danish (35)  support= 0.0408
   21:   Apple Pie (11),  Hot Coffee (45),  Almond Twist (37),   Coffee Eclair (7)   support= 0.0308

Skyline Itemsets:  21

   Rule 1:   Chocolate Cake (0),   Casino Cake (2)   --> Chocolate Coffee (46) [sup= 0.0312  conf= 0.912280701754 ]
   Rule 2:   Apricot Croissant (32),  Hot Coffee (45)   --> Blueberry Tart (16) [sup= 0.0328  conf= 0.942528735632 ]
   Rule 3:   Cherry Tart (18),  Opera Cake (3)    --> Apricot Danish (35) [sup= 0.0408  conf= 0.935779816514 ]
   Rule 4:   Cherry Tart (18),  Apricot Danish (35)  --> Opera Cake (3)    [sup= 0.0408  conf= 0.796875 ]
   Rule 5:   Opera Cake (3), Apricot Danish (35)  --> Cherry Tart (18)     [sup= 0.0408  conf= 0.944444444444 ]
   Rule 6:   Apple Pie (11), Hot Coffee (45),  Almond Twist (37)    --> Coffee Eclair (7)   [sup= 0.0308  conf= 1.0 ]
   Rule 7:   Apple Pie (11), Hot Coffee (45),  Coffee Eclair (7)    --> Almond Twist (37)   [sup= 0.0308  conf= 1.0 ]
   Rule 8:   Apple Pie (11), Almond Twist (37),   Coffee Eclair (7)    --> Hot Coffee (45)     [sup= 0.0308  conf= 0.806282722513 ]
   Rule 9:   Hot Coffee (45),   Almond Twist (37),   Coffee Eclair (7)    --> Apple Pie (11)     [sup= 0.0308  conf= 1.0 ]

==================================================================
Dataset:  data/1000/1000-out1.csv  MinSup:  0.03  MinConf:  0.5
==================================================================

   1:   Berry Tart (14),  Bottled Water (44)   support= 0.034
   2:   Strawberry Cake (4), Napoleon Cake (9)    support= 0.049
   3:   Chocolate Cake (0),  Casino Cake (2)   support= 0.04
   4:   Raspberry Cookie (23),  Lemon Lemonade (40)  support= 0.031
   5:   Marzipan Cookie (27),   Tuile Cookie (28)    support= 0.053
   6:   Blueberry Tart (16), Apricot Croissant (32)  support= 0.04
   7:   Blueberry Tart (16), Hot Coffee (45)   support= 0.033
   8:   Gongolais Cookie (22),  Truffle Cake (5)  support= 0.058
   9:   Cherry Tart (18), Opera Cake (3)    support= 0.041
   10:   Cheese Croissant (33), Orange Juice (42)    support= 0.038
   11:   Raspberry Cookie (23), Lemon Cookie (24)    support= 0.033
   12:   Lemon Cookie (24),  Lemon Lemonade (40)  support= 0.031
   13:   Apricot Croissant (32),   Hot Coffee (45),  Blueberry Tart (16) support= 0.032
   14:   Apple Croissant (31),  Apple Tart (12),  Apple Danish (36),   Cherry Soda (48)   support= 0.031

Skyline Itemsets:  14

   Rule 1:   Strawberry Cake (4)   --> Napoleon Cake (9)    [sup= 0.049  conf= 0.538461538462 ]
   Rule 2:   Napoleon Cake (9)  --> Strawberry Cake (4)     [sup= 0.049  conf= 0.544444444444 ]
   Rule 3:   Casino Cake (2)    --> Chocolate Cake (0)   [sup= 0.04  conf= 0.555555555556 ]
   Rule 4:   Marzipan Cookie (27)  --> Tuile Cookie (28)    [sup= 0.053  conf= 0.588888888889 ]
   Rule 5:   Tuile Cookie (28)  --> Marzipan Cookie (27)    [sup= 0.053  conjf= 0.519607843137 ]
   Rule 6:   Apricot Croissant (32)   --> Blueberry Tart (16)     [sup= 0.04 conf= 0.526315789474 ]
   Rule 7:   Gongolais Cookie (22)    --> Truffle Cake (5)     [sup= 0.058  conf= 0.537037037037 ]
   Rule 8:   Truffle Cake (5)   --> Gongolais Cookie (22)   [sup= 0.058  conf= 0.563106796117 ]
   Rule 9:   Opera Cake (3)  --> Cherry Tart (18)     [sup= 0.041  conf= 0.525641025641 ]
   Rule 10:     Lemon Cookie (24)  --> Raspberry Cookie (23)   [sup= 0.033  conf= 0.5 ]
   Rule 11:     Apricot Croissant (32),  Hot Coffee (45)   --> Blueberry Tart (16)   [sup= 0.032  conf= 1.0 ]
   Rule 12:     Apple Croissant (31), Apple Tart (12),  Apple Danish (36)    --> Cherry Soda (48)   [sup= 0.031  conf= 0.775 ]
   Rule 13:     Apple Croissant (31), Apple Danish (36),   Cherry Soda (48)  --> Apple Tart (12)    [sup= 0.031  conf= 1.0 ]
   Rule 14:     Apple Tart (12),   Apple Danish (36),   Cherry Soda (48)  --> Apple Croissant (31)     [sup= 0.031  conf= 1.0 ]

