(.venv) nathan@Nathans-MacBookPro-6 GSRF Model % .venv/bin/python Model/run_paper_pipeline.py
Preprocessing (paper §4)...
Task 1: GSRF...
Task 2: Logistic...
/Users/nathan/Desktop/University/Columbia/Semesters/2026_Spring/CUMMW/GSRF Model/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
  warnings.warn(
Task 3: Sports...
Task 4: Lasso Great Coach...
Sensitivity...

--- Task 1 Gold (Table 3 style) ---
{'set': 'Training set', 'R2': 0.8665426732021072, 'MAE': 0.6716286014303522, 'RMSE': 1.9977392647193537}
{'set': 'Cross validation set', 'R2': 0.632349097190432, 'MAE': 1.0561075126742474, 'RMSE': 3.315777751839424}
{'set': 'Test set', 'R2': 0.7771979571226375, 'MAE': 0.8888015220490829, 'RMSE': 2.3439687910534137}

--- Task 1 Total (Table 4 style) ---
{'set': 'Training set', 'R2': 0.9066268636207475, 'MAE': 1.4906844453185037, 'RMSE': 4.373046847953523}
{'set': 'Cross validation set', 'R2': 0.7227243945862516, 'MAE': 2.4487324125943633, 'RMSE': 7.53579347246505}
{'set': 'Test set', 'R2': 0.8801405719358407, 'MAE': 1.6744592985557958, 'RMSE': 4.927952430265183}

--- Top 10 Gold 2028 (Table 5 style) ---
Team  PredictedGold2028  PredictedTotal2028  Gold_lower95  Gold_upper95  Total_lower95  Total_upper95
 USA          41.231292          105.961076     34.231292     48.231292      90.961076     120.961076
 FRA          31.711154           75.673377     24.711154     38.711154      60.673377      90.673377
 AUS          28.048115           70.218087     21.048115     35.048115      55.218087      85.218087
 CHN          27.044457           44.641877     20.044457     34.044457      29.641877      59.641877
 GER          26.215446           53.930784     19.215446     33.215446      38.930784      68.930784
 GBR          26.132570           55.898410     19.132570     33.132570      40.898410      70.898410
 ITA          25.927211           43.553553     18.927211     32.927211      28.553553      58.553553
 JPN          14.247933           53.630467      7.247933     21.247933      38.630467      68.630467
 NED          10.688976           33.902472      3.688976     17.688976      18.902472      48.902472
 ESP           9.769456           40.988236      2.769456     16.769456      25.988236      55.988236

--- Top 10 Total 2028 (Table 6 style) ---
Team  PredictedGold2028  PredictedTotal2028  Gold_lower95  Gold_upper95  Total_lower95  Total_upper95
 USA          41.231292          105.961076     34.231292     48.231292      90.961076     120.961076
 FRA          31.711154           75.673377     24.711154     38.711154      60.673377      90.673377
 AUS          28.048115           70.218087     21.048115     35.048115      55.218087      85.218087
 GBR          26.132570           55.898410     19.132570     33.132570      40.898410      70.898410
 GER          26.215446           53.930784     19.215446     33.215446      38.930784      68.930784
 JPN          14.247933           53.630467      7.247933     21.247933      38.630467      68.630467
 CHN          27.044457           44.641877     20.044457     34.044457      29.641877      59.641877
 ITA          25.927211           43.553553     18.927211     32.927211      28.553553      58.553553
 ESP           9.769456           40.988236      2.769456     16.769456      25.988236      55.988236
 NED          10.688976           33.902472      3.688976     17.688976      18.902472      48.902472
 CAN           9.089137           30.730687      2.089137     16.089137      15.730687      45.730687

--- Task 2 Table 11 (p>0.2) ---
Team  a_k  e_k  p_k  MedalWin_Prob2028
 CRT    0    0    1           0.918714
 SAA    0    0    1           0.918714
 WIF    0    0    1           0.918714
 UNK    0    0    1           0.918714
 YMD    0    0    1           0.918714
 ROT    0    0    1           0.918714
 NFL    0    0    1           0.918714
 RHO    0    0    1           0.918714
 NBO    0    0    1           0.918714
 IOA    0    0    4           0.918714
 UAR    0    0    1           0.918714
 MAL    0    0    2           0.918714
 VNM    0    0    6           0.918714
 YAR    0    0    2           0.918714
 ROC    0    0    1           0.918714
 BIZ    1    1   14           0.912536
 NRU    1    1    8           0.912536
 LIE    1    1   19           0.912536
 SOM    1    1   11           0.912536
 ASA    2    2   10           0.905937
 COK    2    2   10           0.905937
 MTN    2    2   11           0.905937
 TUV    2    2    5           0.905937
 MYA    2    2   19           0.905937
 SOL    2    2   11           0.905937
 SWZ    3    3   12           0.898894
 CAM    3    3   11           0.898894
 CHA    3    3   14           0.898894
 SEY    3    3   11           0.898894
 GEQ    3    3   11           0.898894
 BRU    3    3    7           0.898894
 KIR    3    3    6           0.898894
 STP    3    3    8           0.898894
 BHU    3    3   11           0.898894
 PLW    3    3    7           0.898894
 SKN    3    3    8           0.898894
 LES    3    3   13           0.898894
 FSM    3    3    7           0.898894
 MAW    3    3   12           0.898894
 AND    2    4   13           0.893690
 OMA    4    4   11           0.891388
 BOL    4    4   16           0.891388
 VIN    4    4   10           0.891388
 MHL    4    4    5           0.891388
 HON    4    4   13           0.891388
 CGO    4    4   14           0.891388
 CAF    4    4   12           0.891388
 SLE    4    4   13           0.891388
 LAO    4    4   11           0.891388
 TLS    4    4    6           0.891388
 IVB    4    4   11           0.891388
 YEM    4    4    9           0.891388
 MDV    5    4   10           0.890221
 SSD   14    3    3           0.886250
 CAY    4    5   12           0.884627
 COM    4    5    8           0.884627
 BAN    5    5   11           0.883397
 ANT    5    5   12           0.883397
 MLT    5    5   18           0.883397
 BEN    5    5   13           0.883397
 LBR    8    5   14           0.879637
 BIH    5    6    9           0.876208
 VAN    6    6   10           0.874900
 COD    6    6   12           0.874900
 MON    6    6   23           0.874900
 GBS    6    6    8           0.874900
 LBA    6    6   12           0.874900
 NEP    7    6   15           0.873581
 ARU    6    7   10           0.867266
 NCA    7    7   14           0.865879
 MAD    7    7   14           0.865879
 PNG    7    7   12           0.865879
 RWA    7    8   11           0.857783
 GAM    7    8   11           0.857783
 PLE    8    8    8           0.856313
 MLI   24    6   15           0.849284
 GUM    7    9   10           0.849284
 ESA    8    9   14           0.847742
 LBN    9    9    2           0.846187
 GUI   25    7   13           0.838756
 ANG   25   10   11           0.809228

--- Sensitivity ---
athletes_num: 11.0759%; Total event: 0.0000%
(.venv) nathan@Nathans-MacBookPro-6 GSRF Model % 