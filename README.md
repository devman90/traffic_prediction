# traffic_predictions
1. Keras, Tensorflow 설치 필요.
2. 'input' 폴더에 필요한 input 파일들을 위치시킨다.  
 - input/training_data.csv
 - input/test_data.csv
3. 전처리 수행
 - `python preprocess.py` 실행 -> 'preprocessed' 폴더에 전처리된 *.csv 파일 생성  
4. model 학습
 - 완료 시 submission.csv 파일 담긴 모델별 폴더 생성됨, 오래걸림
 - 각 json 파일의 EARLYSTOP을 줄여도 성능 차이는 얼마 안날 수 있음
 - `python train.py --config MODEL1.json`
 - `python train.py --config MODEL2.json`
 - `python train.py --config MODEL3.json`
 - `python train.py --config MODEL4.json`
5. 앙상블 수행
 - `python ensemble.py`