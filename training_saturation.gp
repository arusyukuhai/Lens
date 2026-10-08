set terminal pngcairo size 1500,950 enhanced font 'Arial,11'
set output './training_saturation.png'
set datafile separator comma
set grid xtics ytics y2tics
set key outside horizontal top center
set xlabel 'Completed optimization stages'
set ylabel '-log_2(1-Spearman)'
set y2label 'Reconstruction accuracy (corrupted positions only)'
set y2tics
set yrange [0:*]
set y2range [0:1]
set title 'Lens -- single-path greedy DE / 1-pass Replacer'
# log transforms AFTER raw-score 5-point averaging and running max.
trans(x) = -log( (1.0 - ((x >= 0.999999999999) ? 0.999999999999 : x)) )/log(2.0)
plot './training_history.csv' using 1:(trans(column(6))) with lines lw 1 lc rgb '#999999' title 'Spearman current', \
     '' using 1:(trans(column(7))) with lines lw 2 lc rgb '#157bb8' title 'Spearman MA(5)', \
     '' using 1:((column(1)>=5)?trans(column(8)):1/0) with lines lw 2 lc rgb '#145a32' title 'Best historical MA(5)', \
     '' using 1:9 axes x1y2 with lines lw 1 dt 2 lc rgb '#b0a9a0' title 'Inference current', \
     '' using 1:10 axes x1y2 with lines lw 2 lc rgb '#e67e22' title 'Inference MA(5)', \
     '' using 1:((column(1)>=5)?column(11):1/0) axes x1y2 with lines lw 2 lc rgb '#ba4a00' title 'Inference best MA(5)'
