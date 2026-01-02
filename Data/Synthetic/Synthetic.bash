#!/bin/bash
# SCRIPT TO GENERATE A LIBRARY OF 124,040,000 SEQUENCES

echo -e "Sequence\tC0" > Synthetic.dat
 
# EACH ITERATION = 500,000 SEQUENCES ADDED
for i in {1..20}; do

    # REMOVE PREVIOUS OUTPUT
    rm -f CNN_500k.dat
    
    # RUN USING 5 PARALLEL PROCESSES
    python3 "/Users/prabh/200 RESEARCH/Cluster Expansion Project/Data/Synthetic/Synthetic.py"
    
    # APPEND TEMP FILE TO THE COMBINED FILE
    cat "CNN_500k.dat" >> Synthetic.dat
    
    # UPDATE LOGS FOR MYSELF TO SEE IN TERMINAL
    total_seqs=$((i * 500000))
    mil=$(awk "BEGIN {printf \"%.1f\", $total_seqs/1000000}")
    echo "${mil}M sequences concatendated"
done

