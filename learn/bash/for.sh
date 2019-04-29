for i in hello \
             world\
             'hello world'\
               # some comment
do
    echo $i
done
mexp=testdir
if [ ! -d $mexp ]; then
    mkdir $mexp
else
    echo "$mexp exists."
    echo '$mexp exists.'
fi
