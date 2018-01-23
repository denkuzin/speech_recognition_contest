if [ $1 ];
then
gcloud compute scp $1 gpu-instance-1:~/tensorflow_speech_recognition/
else
gcloud compute scp --recurse ../tensorflow_speech_recognition gpu-instance-1:~/
fi
