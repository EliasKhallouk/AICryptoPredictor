# AICryptoPredictor

date starting : 30/08/2025
last update : 02/09/2025

### File : crontabe -e
00 */1 * * * /home/elias/anaconda3/envs/crypto_predictor/bin/python /home/elias/PROJECT/AICryptoPredictor/notebooks/update_data.py >> /home/elias/PROJECT/AICryptoPredictor/Output/cron_update.log 2>&1
00 */1 * * * /home/elias/anaconda3/envs/crypto_predictor/bin/python /home/elias/PROJECT/AICryptoPredictor/notebooks/pipeline2.py >> /home/elias/PROJECT/AICryptoPredictor/Output/cron_update.log
