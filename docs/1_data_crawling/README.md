## 설명
콘센트로부터 송신되는 전력 데이터를 KSB 프레임워크에서 받아서 처리하는 워크플로우와 Raspberry PI로부터 송신되는 전력 데이터를 HTTP로 송부하기 위한 소스코드가 들어 있습니다.
`데이터 수집시 HTTP 패킷 생성.py` 는 Raspberry PI에서 실행되며, GPIO로 입력된 데이터를 KSB Framework의 엔드포인트로 데이터를 송부합니다.
KSB 프레임워크 상에선 `데이터 수집 워크플로우.json` 파일을 등록하면 자동으로 등록됩니다.

1. KSB 프레임워크에서 HTTP 입력을 mongodb에 저장하는 워크플로우 생성 (센서로부터 데이터 전송 기능 구현은 라즈베리파이에 별도로 구현)
2. 센서 입력 구성 (콘센트 - 수신기 - 라즈베리파이 (수신기부터 JSON 타입임)
3. 라즈베리파이에서 파이썬 기반으로 구성한 serial to http 스크립트 실행
4. 환경 조성 후 화재 환경시 버튼 누르면서 데이터 셋 구성
5. 워크플로우 종료 후 데이터베이스 확인
```
Show dbs
Use Powerdata
show collections
db.Powerdata_from_http.find()
```
6. csv export를 위해 도커 상에서 (mongodb/bin 에서 mongoexport --host localhost --db Powerdata --collection Powerdata_from_http --type=csv --fields _id,error,mode,plug_exist,plug_mode,plug_pow,power_count,powerdata --out ~/ksb-data/data.csv 입력)
7. chmod 777 ~/ksb-data/data.csv
```
[도커 외부에서 docker cp [컨테이너명 - csle1]:/home/csle/data.csv ~/data1.csv ]
```