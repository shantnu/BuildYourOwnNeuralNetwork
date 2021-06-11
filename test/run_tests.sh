set -ex
cp ../mnist_data.zip .
unzip -o mnist_data.zip

for f in *.py; do
python "$f"
done
