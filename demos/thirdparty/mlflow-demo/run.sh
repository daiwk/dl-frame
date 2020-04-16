nohup mlflow ui &


mlflow run ./ -P alpha=0.4 --no-conda

run_id=c7dc4803b062493ba9e003569a30a1b3

nohup mlflow models serve -m runs:/c7dc4803b062493ba9e003569a30a1b3/model --port 3333 --no-conda &

curl -d '{"columns":["fixed acidity","volatile acidity"], "data":[[1,1]]}' -H 'Content-Type: application/json; format=pandas-split' -X POST localhost:3333/invocations


