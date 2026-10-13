# Lens plot styling, based on the classic clean scientific plot.
# ALL four series use the SAME plotted transform: -log2(1 - Spearman rho).
# The underlying TSV remains raw Spearman rho for scientific reproducibility.
set terminal pngcairo size 1600,870 enhanced font ',14'
set output 'training.png'
set datafile separator '\t'
set datafile missing 'nan'
set encoding utf8

# Matplotlib-like white canvas, delicate grid and restrained typography.
set border 3 back linewidth 1.2 linecolor rgb '#323232'
set tics nomirror out scale 0.65 textcolor rgb '#292929'
set grid xtics ytics back linewidth 0.7 linecolor rgb '#dddddd'
set lmargin 11
set rmargin 4
set tmargin 3
set bmargin 5
set title 'Lens / Training progress' font ',18' offset 0,0.4
set xlabel 'iteration' offset 0,0.7
set ylabel '-log2(1 - Spearman rho)' offset 1.0,0

# All y values have the same meaning: never mix raw rho with transformed rho.
# Keep a fixed x origin so even the first iteration has a nonzero x-range.
set xrange [0:*]
set autoscale y
set key left bottom opaque box linecolor rgb '#c4c4c4' linewidth 0.8 samplen 2.5 spacing 1.15

# The protected held-out curve is the primary generalization indicator.
set style line 1 lt 1 lw 2.9 pt 7  ps 0.65 lc rgb '#2b7bb9'
set style line 2 lt 1 lw 2.6 pt 5  ps 0.65 lc rgb '#008f80'
set style line 3 lt 1 lw 2.3 pt 9  ps 0.60 lc rgb '#e68018'
set style line 4 lt 1 lw 2.0 pt 13 ps 0.55 lc rgb '#8855a2'

# This is log BASE TWO (not the natural logarithm). For rho == 1,
# cap the visualization only, preventing Inf from blanking the plot.
# Negative Spearman rho is valid and maps to a negative plotted value.
f(r) = -log(1.0 - (r >= 1.0 ? 1.0 - 1.0e-12 : r)) / log(2.0)

# training_v3.tsv: 1 iteration, 3 search rho, 4 acceptance rho,
#                  5 anchor rho, 6 held-out rho.
plot 'training_v3.tsv' using 1:(f($6)) with linespoints ls 1 title 'Held-out (fixed)', \
     '' using 1:(f($5)) with linespoints ls 2 title 'Anchor (rotating)', \
     '' using 1:(f($4)) with linespoints ls 3 title 'Acceptance (rotating)', \
     '' using 1:(f($3)) with linespoints ls 4 title 'Search validation'
unset output
