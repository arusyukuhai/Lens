set terminal pngcairo size 1450,820 enhanced font 'Arial,13'
set output 'training.png'
set datafile separator whitespace
set border lc rgb '#454545' lw 1.2
set tics textcolor rgb '#333333'
set grid xtics ytics lc rgb '#dddddd' lw 1
set key top left opaque box
set xlabel 'Step'
set ylabel '-ln(1 - Pearson)'
set title 'Lens | Ridge count-rule search'
set style line 1 lc rgb '#a6bee9' lw 1.3 pt 7 ps 0.30
set style line 2 lc rgb '#a2d9bf' lw 1.3 pt 5 ps 0.30
set style line 3 lc rgb '#3265c9' lw 3.0
set style line 4 lc rgb '#189567' lw 3.0
plot 'training.dat' using 1:2 with linespoints ls 1 title 'Train (raw)', \
     'training.dat' using 1:3 with linespoints ls 2 title 'Held-out (raw)', \
     'training.dat' using 1:8 with lines ls 3 title 'Train (Pearson MA75)', \
     'training.dat' using 1:9 with lines ls 4 title 'Held-out (Pearson MA75)'
