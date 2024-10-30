ENVS_DIR="rtrend_tools/envs"
FLEX_FNAME="latest.yml"
EXACT_FNAME="explicit.txt"


while true; do
    read -p "This will overwrite current environment snapshots. Proceed? (y/n) " yn
    case $yn in
        [Yy]* ) 
            echo Exporting flexible file
            conda env export --from-history > $ENVS_DIR/$FLEX_FNAME  # Export the "flexible" snapshot
            echo Exporting exact file
            conda list --explicit > $ENVS_DIR/$EXACT_FNAME  # Export the "explicit" snapshot
            break;;
        [Nn]* ) echo "Quitting..."; exit;;
        * ) echo "Please answer y or n.";;
    esac
done
