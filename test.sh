tfmdp Res10-v4.rddl -l 1024 -iln -a elu -est pd -b 128 -hr 40 -e 200 -lr 0.001 -v -ld /tmp/tfmdp/res10-v4/layers=1024/est=pd
tfmdp Res10-v4.rddl -l 1024 -iln -a elu -est sf -b 128 -hr 40 -e 200 -lr 0.001 -v -ld /tmp/tfmdp/res10-v4/layers=1024/est=sf
tfmdp Res10-v4.rddl -l 1024 -iln -a elu -est hybrid --n-step 2  -b 128 -hr 40 -e 200 -lr 0.001 -v -ld /tmp/tfmdp/res10-v4/layers=1024/n-step=2
tfmdp Res10-v4.rddl -l 1024 -iln -a elu -est hybrid --n-step 4  -b 128 -hr 40 -e 200 -lr 0.001 -v -ld /tmp/tfmdp/res10-v4/layers=1024/n-step=4
tfmdp Res10-v4.rddl -l 1024 -iln -a elu -est hybrid --n-step 8  -b 128 -hr 40 -e 200 -lr 0.001 -v -ld /tmp/tfmdp/res10-v4/layers=1024/n-step=8
tfmdp Res10-v4.rddl -l 1024 -iln -a elu -est hybrid --n-step 10 -b 128 -hr 40 -e 200 -lr 0.001 -v -ld /tmp/tfmdp/res10-v4/layers=1024/n-step=10
tfmdp Res10-v4.rddl -l 1024 -iln -a elu -est hybrid --n-step 20 -b 128 -hr 40 -e 200 -lr 0.001 -v -ld /tmp/tfmdp/res10-v4/layers=1024/n-step=20
tfmdp Res10-v4.rddl -l 1024 -iln -a elu -est hybrid --n-step 30 -b 128 -hr 40 -e 200 -lr 0.001 -v -ld /tmp/tfmdp/res10-v4/layers=1024/n-step=30



# tfmdp Res8-v1.rddl -l 1024 -iln -a elu -est pd -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res8-v1/layers=1024/est=pd
# tfmdp Res8-v1.rddl -l 1024 -iln -a elu -est sf -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res8-v1/layers=1024/est=sf
# tfmdp Res8-v1.rddl -l 1024 -iln -a elu -est hybrid --n-step 2  -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res8-v1/layers=1024/n-step=2
# tfmdp Res8-v1.rddl -l 1024 -iln -a elu -est hybrid --n-step 4  -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res8-v1/layers=1024/n-step=4
# tfmdp Res8-v1.rddl -l 1024 -iln -a elu -est hybrid --n-step 8  -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res8-v1/layers=1024/n-step=8
# tfmdp Res8-v1.rddl -l 1024 -iln -a elu -est hybrid --n-step 10 -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res8-v1/layers=1024/n-step=10
# tfmdp Res8-v1.rddl -l 1024 -iln -a elu -est hybrid --n-step 20 -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res8-v1/layers=1024/n-step=20
# tfmdp Res8-v1.rddl -l 1024 -iln -a elu -est hybrid --n-step 30 -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res8-v1/layers=1024/n-step=30



# tfmdp Res10-v3.rddl -l 1024 -iln -a elu -est pd -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res10-v3/layers=1024/est=pd
# tfmdp Res10-v3.rddl -l 1024 -iln -a elu -est sf -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res10-v3/layers=1024/est=sf
# tfmdp Res10-v3.rddl -l 1024 -iln -a elu -est hybrid --n-step 2  -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res10-v3/layers=1024/n-step=2
# tfmdp Res10-v3.rddl -l 1024 -iln -a elu -est hybrid --n-step 4  -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res10-v3/layers=1024/n-step=4
# tfmdp Res10-v3.rddl -l 1024 -iln -a elu -est hybrid --n-step 8  -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res10-v3/layers=1024/n-step=8
# tfmdp Res10-v3.rddl -l 1024 -iln -a elu -est hybrid --n-step 10 -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res10-v3/layers=1024/n-step=10
# tfmdp Res10-v3.rddl -l 1024 -iln -a elu -est hybrid --n-step 20 -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res10-v3/layers=1024/n-step=20
# tfmdp Res10-v3.rddl -l 1024 -iln -a elu -est hybrid --n-step 30 -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res10-v3/layers=1024/n-step=30


# tfmdp Res10-v0.rddl -l 1024 -iln -a elu -est pd -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res10-v0/layers=1024/est=pd
# tfmdp Res10-v0.rddl -l 1024 -iln -a elu -est sf -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res10-v0/layers=1024/est=sf
# tfmdp Res10-v0.rddl -l 1024 -iln -a elu -est hybrid --n-step 2  -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res10-v0/layers=1024/n-step=2
# tfmdp Res10-v0.rddl -l 1024 -iln -a elu -est hybrid --n-step 4  -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res10-v0/layers=1024/n-step=4
# tfmdp Res10-v0.rddl -l 1024 -iln -a elu -est hybrid --n-step 8  -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res10-v0/layers=1024/n-step=8
# tfmdp Res10-v0.rddl -l 1024 -iln -a elu -est hybrid --n-step 10 -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res10-v0/layers=1024/n-step=10
# tfmdp Res10-v0.rddl -l 1024 -iln -a elu -est hybrid --n-step 20 -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res10-v0/layers=1024/n-step=20
# tfmdp Res10-v0.rddl -l 1024 -iln -a elu -est hybrid --n-step 30 -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res10-v0/layers=1024/n-step=30



# tfmdp Res8-v3.rddl -l 1024 -iln -a elu -est pd -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res8-v3/layers=1024/est=pd
# tfmdp Res8-v3.rddl -l 1024 -iln -a elu -est sf -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res8-v3/layers=1024/est=sf
# # tfmdp Res8-v3.rddl -l 1024 -iln -a elu -est hybrid --n-step 1  -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res8-v3/layers=1024/n-step=1
# tfmdp Res8-v3.rddl -l 1024 -iln -a elu -est hybrid --n-step 2  -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res8-v3/layers=1024/n-step=2
# tfmdp Res8-v3.rddl -l 1024 -iln -a elu -est hybrid --n-step 4  -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res8-v3/layers=1024/n-step=4
# tfmdp Res8-v3.rddl -l 1024 -iln -a elu -est hybrid --n-step 8  -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res8-v3/layers=1024/n-step=8
# tfmdp Res8-v3.rddl -l 1024 -iln -a elu -est hybrid --n-step 10 -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res8-v3/layers=1024/n-step=10
# tfmdp Res8-v3.rddl -l 1024 -iln -a elu -est hybrid --n-step 20 -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res8-v3/layers=1024/n-step=20
# tfmdp Res8-v3.rddl -l 1024 -iln -a elu -est hybrid --n-step 30 -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res8-v3/layers=1024/n-step=30
# tfmdp Res8-v3.rddl -l 1024 -iln -a elu -est hybrid --n-step 40 -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res8-v3/layers=1024/n-step=40
# tfmdp Res8-v3.rddl -l 1024 -iln -a elu -est hybrid --n-step 42 -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res8-v3/layers=1024/n-step=42


# tfmdp Res8-v0.rddl -l 1024 -iln -a elu -est pd -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res8-v0/layers=1024/est=pd
# tfmdp Res8-v0.rddl -l 1024 -iln -a elu -est sf -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res8-v0/layers=1024/est=sf
# # tfmdp Res8-v0.rddl -l 1024 -iln -a elu -est hybrid --n-step 1  -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res8-v0/layers=1024/n-step=1
# tfmdp Res8-v0.rddl -l 1024 -iln -a elu -est hybrid --n-step 2  -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res8-v0/layers=1024/n-step=2
# tfmdp Res8-v0.rddl -l 1024 -iln -a elu -est hybrid --n-step 4  -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res8-v0/layers=1024/n-step=4
# tfmdp Res8-v0.rddl -l 1024 -iln -a elu -est hybrid --n-step 8  -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res8-v0/layers=1024/n-step=8
# tfmdp Res8-v0.rddl -l 1024 -iln -a elu -est hybrid --n-step 10 -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res8-v0/layers=1024/n-step=10
# tfmdp Res8-v0.rddl -l 1024 -iln -a elu -est hybrid --n-step 20 -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res8-v0/layers=1024/n-step=20
# tfmdp Res8-v0.rddl -l 1024 -iln -a elu -est hybrid --n-step 30 -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res8-v0/layers=1024/n-step=30
# tfmdp Res8-v0.rddl -l 1024 -iln -a elu -est hybrid --n-step 40 -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res8-v0/layers=1024/n-step=40
# tfmdp Res8-v0.rddl -l 1024 -iln -a elu -est hybrid --n-step 42 -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res8-v0/layers=1024/n-step=42



# tfmdp Reservoir-8 -l 256 128 64 32 -iln -a elu -est sf -b 128 -hr 40 -e 200 -lr 0.001 -v -ld /tmp/tfmdp/res8/layers=256+128+64+32/est=sf
# tfmdp Reservoir-8 -l 256 128 64 32 -iln -a elu -est pd -b 128 -hr 40 -e 200 -lr 0.001 -v -ld /tmp/tfmdp/res8/layers=256+128+64+32/est=pd


# tfmdp HVAC-3 -l 1024 -iln -a elu -est pd -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/hvac3/pd
# tfmdp HVAC-3 -l 1024 -iln -a elu -est sf -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/hvac3/sf
# tfmdp HVAC-3 -l 1024 -iln -a elu -est sf --baseline -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/hvac3/sf+baseline


# tfmdp Navigation-v3 -l 1024 -a elu -est pd -b 128 -hr 20 -e 50 -lr 0.001 -v -ld /tmp/tfmdp/nav2/pd
# tfmdp Navigation-v3 -l 1024 -a elu -est sf -b 128 -hr 20 -e 50 -lr 0.001 -v -ld /tmp/tfmdp/nav2/sf
# tfmdp Navigation-v3 -l 1024 -a elu -est sf --baseline --n-step 1  -b 128 -hr 20 -e 50 -lr 0.001 -v -ld /tmp/tfmdp/nav2/sf+baseline
# tfmdp Navigation-v3 -l 1024 -a elu -est hybrid --n-step 4  -b 128 -hr 20 -e 50 -lr 0.001 -v -ld /tmp/tfmdp/nav2/hybrid/n-step=4
# tfmdp Navigation-v3 -l 1024 -a elu -est hybrid --n-step 4 --baseline -b 128 -hr 20 -e 50 -lr 0.001 -v -ld /tmp/tfmdp/nav2/hybrid+baseline/n-step=4
# tfmdp Navigation-v3 -l 1024 -a elu -est hybrid --n-step 10  -b 128 -hr 20 -e 50 -lr 0.001 -v -ld /tmp/tfmdp/nav2/hybrid/n-step=10
# tfmdp Navigation-v3 -l 1024 -a elu -est hybrid --n-step 10 --baseline -b 128 -hr 20 -e 50 -lr 0.001 -v -ld /tmp/tfmdp/nav2/hybrid+baseline/n-step=10
# tfmdp Navigation-v3 -l 1024 -a elu -est hybrid --n-step 20  -b 128 -hr 20 -e 50 -lr 0.001 -v -ld /tmp/tfmdp/nav2/hybrid/n-step=20
# tfmdp Navigation-v3 -l 1024 -a elu -est hybrid --n-step 20 --baseline -b 128 -hr 20 -e 50 -lr 0.001 -v -ld /tmp/tfmdp/nav2/hybrid+baseline/n-step=20


# tfmdp Reservoir-20 -l 1024 -iln -a elu -est pd -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res20/pd
# tfmdp Reservoir-20 -l 1024 -iln -a elu -est sf -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res20/sf
# tfmdp Reservoir-20 -l 1024 -iln -a elu -est sf --baseline --n-step 1  -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res20/sf+baseline
# tfmdp Reservoir-20 -l 1024 -iln -a elu -est hybrid --n-step 4  -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res20/hybrid/n-step=4
# tfmdp Reservoir-20 -l 1024 -iln -a elu -est hybrid --n-step 4 --baseline -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res20/hybrid+baseline/n-step=4
# tfmdp Reservoir-20 -l 1024 -iln -a elu -est hybrid --n-step 10  -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res20/hybrid/n-step=10
# tfmdp Reservoir-20 -l 1024 -iln -a elu -est hybrid --n-step 10 --baseline -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res20/hybrid+baseline/n-step=10
# tfmdp Reservoir-20 -l 1024 -iln -a elu -est hybrid --n-step 20  -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res20/hybrid/n-step=20
# tfmdp Reservoir-20 -l 1024 -iln -a elu -est hybrid --n-step 20 --baseline -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res20/hybrid+baseline/n-step=20



# tfmdp Reservoir-8 -l 1024 -iln -a elu -est hybrid --n-step 1  -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res8/layers=1024/n-step=1
# tfmdp Reservoir-8 -l 1024 -iln -a elu -est hybrid --n-step 2  -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res8/layers=1024/n-step=2
# tfmdp Reservoir-8 -l 1024 -iln -a elu -est hybrid --n-step 4  -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res8/layers=1024/n-step=4
# tfmdp Reservoir-8 -l 1024 -iln -a elu -est hybrid --n-step 8  -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res8/layers=1024/n-step=8
# tfmdp Reservoir-8 -l 1024 -iln -a elu -est hybrid --n-step 10 -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res8/layers=1024/n-step=10
# tfmdp Reservoir-8 -l 1024 -iln -a elu -est hybrid --n-step 20 -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res8/layers=1024/n-step=20
# tfmdp Reservoir-8 -l 1024 -iln -a elu -est hybrid --n-step 30 -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res8/layers=1024/n-step=30
# tfmdp Reservoir-8 -l 1024 -iln -a elu -est hybrid --n-step 40 -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res8/layers=1024/n-step=40
# tfmdp Reservoir-8 -l 1024 -iln -a elu -est hybrid --n-step 42 -b 128 -hr 40 -e 100 -lr 0.001 -v -ld /tmp/tfmdp/res8/layers=1024/n-step=42


# tfmdp Reservoir-8 -l 1024 -iln -a elu -est sf -b 128 -hr 40 -e 200 -lr 0.001 -v -ld /tmp/tfmdp/res8/layers=1024/est=sf
# tfmdp Reservoir-8 -l 1024 -iln -a elu -est pd -b 128 -hr 40 -e 200 -lr 0.001 -v -ld /tmp/tfmdp/res8/layers=1024/est=pd
# tfmdp Reservoir-8 -l 256 128 64 32 -iln -a elu -est sf -b 128 -hr 40 -e 200 -lr 0.001 -v -ld /tmp/tfmdp/res8/layers=256+128+64+32/est=sf
# tfmdp Reservoir-8 -l 256 128 64 32 -iln -a elu -est pd -b 128 -hr 40 -e 200 -lr 0.001 -v -ld /tmp/tfmdp/res8/layers=256+128+64+32/est=pd

# tfmdp Navigation-v2 -l 1024 -a elu -est sf -b 128 -hr 20 -e 200 -lr 0.001 -v -ld /tmp/tfmdp/nav2/layers=1024/est=sf
# tfmdp Navigation-v2 -l 1024 -a elu -est pd -b 128 -hr 20 -e 200 -lr 0.001 -v -ld /tmp/tfmdp/nav2/layers=1024/est=pd
# tfmdp Navigation-v2 -l 256 128 64 32 -a elu -est sf -b 128 -hr 20 -e 200 -lr 0.001 -v -ld /tmp/tfmdp/nav2/layers=256+128+64+32/est=sf
# tfmdp Navigation-v2 -l 256 128 64 32 -a elu -est pd -b 128 -hr 20 -e 200 -lr 0.001 -v -ld /tmp/tfmdp/nav2/layers=256+128+64+32/est=pd