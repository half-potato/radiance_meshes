for i in output/*_$1; do echo $i; python scripts/timediff.py ${i}/transform.txt ${i}/ckpt.ply; done
for i in output/*_$1; do echo $i; ls -lh --block-size=M ${i}/ckpt.ply; done
for i in output/*_$1; do echo $i; head -n 8 ${i}/ckpt.ply | tail -n 1; done
