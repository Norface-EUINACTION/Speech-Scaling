# Declare an array of string with type
# declare -a StringArray=("eo" "ca" "et" "eu" "fi" "id" "ka" "ko" "lt" "no" "th" "he" "hu" "tr")
# declare -a StringArray=("en" "de" "fr" "pt" "fi")
declare -a StringArray=("hu")
 
# Iterate the string array using for loop
for val in ${StringArray[@]}; do
   echo "Downloading and unpacking fastText vectors for $val"
   wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.$val.300.vec.gz
   gunzip cc.$val.300.vec.gz
done
