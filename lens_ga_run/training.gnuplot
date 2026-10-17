set terminal pngcairo size 1550,820 enhanced font 'Arial,12'
set output 'training.png'
set xlabel 'Generation'
set ylabel '-ln(1 - Pearson)'
set title 'Lens | GA curves with trailing MA (75 evaluated points)'
set grid
set key outside right top
set style line 1 lc rgb '#4477AA'
set style line 2 lc rgb '#CC6677'
set style line 3 lc rgb '#228833'
set style line 4 lc rgb '#AA7744'
plot 'training.dat' using 1:2 with lines ls 1 dt 3 lw 1 title 'Train raw', \
     'training_ma.dat' using 1:2 with lines ls 1 lw 2.8 title 'Train MA', \
     'training.dat' using 1:3 with lines ls 2 dt 3 lw 1 title 'Held-out raw', \
     'training_ma.dat' using 1:3 with lines ls 2 lw 2.8 title 'Held-out MA', \
     'training.dat' using 1:8 with lines ls 3 dt 3 lw 1 title 'Rotating LOO raw', \
     'training_ma.dat' using 1:4 with lines ls 3 lw 2.8 title 'Rotating LOO MA', \
     'training.dat' using 1:10 with lines ls 4 dt 3 lw 1 title 'Fixed reference raw', \
     'training_ma.dat' using 1:5 with lines ls 4 lw 2.8 title 'Fixed reference MA'
