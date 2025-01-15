cat $1 | sed -n '2p' |awk -F',' '{for(i=1; i<=NF; i++) print $i}' > $2 
