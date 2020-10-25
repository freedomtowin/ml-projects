docker build --tag serving_mobilenet ./docker_container
docker run -d --name serving_base serving_mobilenet
docker cp mobilenet_final serving_base:/models/mobilenet_final/001
docker commit --change "ENV MODEL_NAME mobilenet_final" serving_base serving_mobilenet
docker kill serving_base
docker run -p 8500:8500 -p 8501:8501 -t serving_mobilenet