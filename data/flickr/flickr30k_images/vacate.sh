uniq train/train.txt train/train_clean.txt
for fileName in `cat train/train_clean.txt`; do mv $fileName train; done

uniq test/test.txt test/test_clean.txt
for fileName in `cat test/test_clean.txt`; do mv $fileName test; done


uniq val/val.txt val/val_clean.txt
for fileName in `cat val/val_clean.txt`; do mv $fileName val; done