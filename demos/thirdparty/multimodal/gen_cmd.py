#encoding=utf8
"""gen_cmds"""
import sys
prefix = sys.argv[1]
file_num = int(sys.argv[2])

for i in xrange(0, file_num):
    if i < 10:
        xi = "0" + str(i)
    else:
        xi = str(i)
    x = "cat ./output/%s_%s| ~/daiwenkai/tools/python-2.7.14/bin/python ./read_gcms.py ./output/info.res.%s %s &" \
        % (prefix, xi, xi, "imgs")
    print x
print "wait"
print "cat ./output/info.res.* > info.res"
