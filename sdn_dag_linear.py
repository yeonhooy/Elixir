#!/usr/bin/python
from mininet.net import Mininet
from mininet.node import UserSwitch, OVSKernelSwitch, Controller, Node, OVSSwitch
from mininet.topo import Topo
from mininet.log import lg, info
from mininet.util import irange, quietRun, dumpNodeConnections, dumpPorts

from mininet.link import TCLink
from mininet.node import RemoteController
from mininet.cli import CLI


#import numpy 
import sys,os
import time
import cmd
import random
from functools import partial
flush = sys.stdout.flush

class LinearTopo (Topo):
    "Linear topology of k switches, with one host per switch."
    def __init__(self, k=1, h=1 ,**opts):
        """Init.
            k: number of switches (and hosts)
            hconf: host configuration options
            lconf: link configuration options"""
        super(LinearTopo, self).__init__(**opts)
        self.k = k
        self.h = h
        lastSwitch = None
        #edgeSwitch
        inSwitch = None
        egSwitch = None
        #make siwtch
        for i in irange(1, k):
            #host = self.addHost('h%s' % i)
            switch = self.addSwitch('s%s' % i)
            if i==1:
                inSwitch = switch
            if i==k:
                egSwitch = switch 
            #self.addLink( host, switch)
            if lastSwitch:
                self.addLink( switch, lastSwitch)
            lastSwitch = switch
        
        #make host    
        for j in irange(1, h):
            host = self.addHost('h%s' %j)
            self.addLink(host,inSwitch)
        for l in irange(h+1, 2*h):
            host = self.addHost('h%s' %l)    
            self.addLink(host,egSwitch)

        
    
def manageContainer(ip):
    print('Restart Container')
    os.system('sudo sh onos_CC.sh -t 1 -i %s' %ip)
    time.sleep(1)
    print('maybe done')
    
def startNetHogs(num):
    startTime = 60*60*24*30
    print('***********Start nethogs')
    startTime = time.time()
    os.system("nohup ./nethogsCC.sh %s &" %num)
    #sys.exit()
    time.sleep(3)
    print("next step>>")
    return startTime
    
def quitNetHogs():
    quitTime = 60*60*24*30
    print('***********quit nethogs')
    quitTime = time.time()
    time.sleep(1)
    #os.system('sudo kill -9 -ef nethogs')
    os.system("sudo killall -9 nethogs")
    time.sleep(1)
    return quitTime
def startShark():
    startTime = 60 * 60 * 24 * 30
    print('***********Start wireshark')
    startTime = time.time()
    #os.system("nohup ./nethogsCC.sh %s &" % num)
    os.system("sudo rm -r alphaSDN.pcap")
    os.system("sudo touch alphaSDN.pcap")
    os.system("nohup sudo tshark -i docker0 -w alphaSDN.pcap &")
    #sys.exit()
    time.sleep(2)
    print("next step>>")
    return startTime
    
def stopShark():
    # quitTime = 60*60*24*30
    print('***********quit wireshark')
    # quitTime = time.time()
    time.sleep(1)
    # os.system('sudo kill -9 -ef nethogs')
    os.system("sudo killall -9 tshark")
    time.sleep(1)
    
def readShark(exp,cip,prov):
    if prov == "OpenFlow10":
        prov = "openflow_v1"
    elif prov == "OpenFlow13":
        prov = "openflow_v4"
    print("***** Read and cut wireshark.pacap")
    os.system("sh readShark.sh %s %s %s" %(exp,cip,prov))    

    
def testLoop(s,h,m,l,cip,prov): #i=swich j=host m=connection l=interval
    "Create and test a simple network"
    c = RemoteController('c1',ip=cip,port=10000)
    sw = partial(OVSSwitch, protocols=prov)
    print("create linear")
    topo = LinearTopo(s,h)
    net = Mininet(topo=topo, build=False, switch=sw)
    net.addController(c)
    net.start()
    print "Dumping host connections"
    dumpNodeConnections(net.hosts)
    print "Dumping port"
    dumpPorts(net.switches)
    #print "Testing network connectivity"
    #net.pingAll()
    
    count = 1
    port = 1000
    for i in irange(1,h):
        if count > m:
            break;
        for j in irange(1,h):
            #globals()['s{}'.format(i)] = net.get("h"+str(i/m + 1))
            #globals()['d{}'.format(j)] = net.get("h"+str(j/m + 1 + m))	
            if count > m:
                break;
            src = "h" + str(i)
            dst = "h" + str(j+h)
            print("src: "+src, "/ dst: "+dst)
            s1 = net.get(src)
            d1 = net.get(dst)
            port = port + 1
            print("Generate Traffic bewteen %s and %s using port(%s)" %(d1.params['ip'],s1.params['ip'],port))
            result1 = d1.cmd("iperf3 -s -1 -p %s >> logResult/iperfResult/host1.txt &" % port)
            result2 = s1.cmd("iperf3 -c %s -p %s -n 1 -l 100 >> logResult/iperfResult/host2.txt &" % (d1.params['ip'],port))
            print("traffic connection %s/%s" %(count,m))
            time.sleep(l)
            count=count+1
    
    print("complete!")
    time.sleep(3)
    #CLI(net)
    net.stop()
    
    
def clearPreviousTopo():
    print("**********claer previous Topology *************")
    os.system('sudo mn -c')
    #print("************end mn-c")

if __name__ == '__main__':
    lg.setLogLevel( 'info' )

    #
    #info( "*** Running linearBandwidthTest", sizes, '\n' )
    #linearBandwidthTest( sizes  )
    #exp=652
    #i = switch / 1 ~ 32
    #j=1 #j = host  / 1 ~ 10
    #m=1 #m = connection /1 ~ j*j
    #l=0 #l = inteval / 0~10
    startTime = 60*60*24*30
    quitTime = 60*60*24*30
    resultTime = 60*60
    limitHost = 1
    newH = 10
    ip='172.17.0.1'
    protocol='OpenFlow10'
    cList = [0,0,0]

    #manageContainer()
    exp=1;
    for i in irange(1,3000):
    	#switch 1~64 , host 1~ 10 per each switch 
        newS = random.randrange(1,65)  
        #newS = numpy.random.randint(low=1,high=65,size=1)
        if newH == 10:
        	limitHost = 1
        else:
        	limitHost = newH+1 

        newH = random.randrange(limitHost,11)
        
        if newH ==1:
            newC = 1
        else:
            conn = newH * newH +1
            newC = random.randrange(1,conn)
            print(newC)
        
        if newC > 66:
        	print("67~100 > 1")
        	cList[2] = random.randrange(67,conn)
        	print("34~66 > 1")
        	cList[1] = random.randrange(34,67)
        	print("1~33 > 1")
        	cList[0] = random.randrange(1,34)
        
        elif newC > 33:
        	cList[2]=0
        	print("34~66 > 1")
        	cList[1] = random.randrange(34,conn)
        	print("1~33 > 1")
        	cList[0] = random.randrange(1,34)
        
        else:
        	cList[2]=0
        	cList[1]=0
        	print("1~newC")
        	cList[0] = random.randrange(1,conn)

        for y in irange(0,2):
        	print("s[%s]/h[%s]/c[%s/%s]" %(newS,newH,y+1,cList[y]))
        	if cList[y]!=0:
	        	print("cList[%s] = %s" %(y,cList[y]))
	        	newC = cList[y]
		        newI = random.randrange(0,11)           
		        clearPreviousTopo()
		        manageContainer(ip)
		        print("Exp(%s): <Switch(%s)>, <Host(%s)>, <Connection(%s)>, <Interval(%s)>  " % (exp,newS,newH,newC,newI)) 
		        startShark()
		        testLoop(newS,newH,newC,newI,ip,protocol)
		        stopShark()
		        #sresultTime=quitTime-startTime  
		        time.sleep(5)
		        #f=open("/home/vs/yyh/output/output_%s.txt" % exp , "a")
		        #f.write("\n:6653 %s %s" %(resultTime,resultTime))
		        #f.write("\n:6653 %s %s %s %s" %(i,j,m,l))
		        #f.close()
		        print("parsing result file >> ")
		        #os.system("sudo sh cutNethogs.sh %s" %exp)
		        #print("cut!")
		        readShark(exp,ip,protocol)
		        f1=open("/home/vs/yeonhooy/wireshark_result/totalResult/duration_%s.txt" %exp)
		        f2=open("/home/vs/yeonhooy/wireshark_result/totalResult/totalSendMsg_%s.txt" %exp)
		        f3=open("/home/vs/yeonhooy/wireshark_result/totalResult/totalSendBytes_%s.txt" %exp)
		        f5=open("/home/vs/yeonhooy/wireshark_result/sendFromCon/XsendParseMsg_%s.txt" %exp)
		        f6=open("/home/vs/yeonhooy/wireshark_result/sendFromCon/XsendParseByte_%s.txt" %exp)
		     
		        resultTime=f1.readline()
		        resultTime=float(resultTime)
		        #resultTime = round(resultTime,3)
		        print(resultTime)
		        totSentMsg = f2.readline()
		        totSentMsg = float(totSentMsg)
		        avgSentMsg = totSentMsg/resultTime
		        avgSentMsg = round(avgSentMsg,2)
		        totSendByte = f3.readline()
		        avgSendByte = float(totSendByte)/resultTime/1024
		        avgSendByte = round(avgSendByte,2)
		        maxSendMsg = f5.readline()
		        maxSendMsg = float(maxSendMsg)
		        secMaxSendMsg = maxSendMsg
		        maxSendByte = f6.readline()
		        secMaxSendByte = float(maxSendByte) / 1024
		        secMaxSendByte = round(secMaxSendByte,2)
		        
		        f7=open("/home/vs/yeonhooy/wireshark_result/totalResult/totalRecvMsg_%s.txt" %exp)
		        f8=open("/home/vs/yeonhooy/wireshark_result/totalResult/totalRecvBytes_%s.txt" %exp)
		        f9=open("/home/vs/yeonhooy/wireshark_result/recvByCon/XrecvParseMsg_%s.txt" %exp)
		        f11=open("/home/vs/yeonhooy/wireshark_result/recvByCon/XrecvParseByte_%s.txt" %exp)
		     
		       
		        totRecvMsg = f7.readline()
		        totRecvMsg = float(totRecvMsg)
		        avgRecvMsg = totRecvMsg/resultTime
		        avgRecvMsg = round(avgRecvMsg,2)
		        totRecvByte = f8.readline()
		        avgRecvByte = float(totRecvByte)/resultTime/1024
		        avgRecvByte = round(avgRecvByte,2)
		        maxRecvMsg = f9.readline()
		        maxRecvMsg = float(maxRecvMsg)
		        secMaxRecvMsg = maxRecvMsg 
		        maxRecvByte = f11.readline()
		        secMaxRecvByte = float(maxRecvByte) / 1024
		        secMaxRecvByte = round(secMaxRecvByte,2)
		        #ssent = f7.read()
		        #ssum = float(ssent)
		        #srecv = f8.read()
		        #rsum = float(srecv)
		        
		        #Linear input data rewrite
		        if newS<2:
		            edgeS=newS
		        else:
		            edgeS = 2
		        coreS=newS-edgeS
		        totalH=newH*edgeS
		        totLink=newS-1+totalH
		        newHop= newS
		        
		        print("Exp(%s): <EdgeSwitch(%s)>, <CoreSwitch(%s)>, <Total_Host(%s)>, <Total_link(%s)>, <Hop(%s)>, <Connection(%s)>, <Interval(%s)>  " % (exp, edgeS,coreS,totalH,totLink,newHop,newC,newI))
		        f4=open("/home/vs/yyh/data_linear.txt","a")
		        f4.write("%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s \n" %(edgeS,coreS,totalH,newC,newI,totLink,newHop,resultTime,avgSentMsg,avgSendByte,secMaxSendMsg,secMaxSendByte,avgRecvMsg,avgRecvByte,secMaxRecvMsg,secMaxRecvByte))                
		        exp = exp + 1
		        f1.close()
		        f2.close()
		        f3.close()
		        f4.close()
		        f5.close()
		        f6.close()
		        f7.close()
		        f8.close()
		        f9.close()
		        f11.close()
		        os.system("sudo killall -9 tshark")
	                             
	     
	    #clearPreviousTopo()
	    #iperfTest()
