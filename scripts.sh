while getopts 'm:' OPTION; do
case "$OPTION" in
    m)
        array=("$@")
        array=(${array[@]/${1}})
        array=(${array[@]/${2}})
        array=(${array[@]/${3}})
        array=(${array[@]/${4}})

        python get_data.py $OPTARG ${3} ${4} "${array[@]}"

        read -p "Press any key to exit:" 
        exit 0
        ;;
    ?)
        echo "utilisation de la commande : scripts.sh [-m (kmeans, knn, classification)] [arguments]"
        read -p "Press any key to exit:" 
        exit 1
        ;;
    esac
done


