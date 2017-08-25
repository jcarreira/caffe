git clone 
git clone https://github.com/jcarreira/cirrus.git 

# we need to replace all calls to LOG to something else
# because caffe uses a #define LOG
# ¯\_(ツ)_/¯
cd cirrus/src
find . -type f -print0 | xargs -0 sed -i 's/LOG/LOG_CIRRUS/g'
cd ../tests
find . -type f -print0 | xargs -0 sed -i 's/LOG/LOG_CIRRUS/g'
cd ../

if [ -d examples ] then
  cd examples
  find . -type f -print0 | xargs -0 sed -i 's/LOG/LOG_CIRRUS/g'
  cd ..
fi

./bootstrap.sh
make
