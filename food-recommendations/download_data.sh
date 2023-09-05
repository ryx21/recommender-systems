dir="data"
mkdir $dir
#kaggle datasets download -d shuyangli94/food-com-recipes-and-user-interactions -p $dir
unzip "${dir}/food-com-recipes-and-user-interactions.zip" -d $dir