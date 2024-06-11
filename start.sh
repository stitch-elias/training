  #!/bin/sh

	logFile="./stop.log"
	echo "TWS service started by PID $$" >> "$logFile"

    while :


    do

    stillRunning=$(ps -ef |grep "./CardRecognition ../etc/config.ini" |grep -v "grep")

    if [ "$stillRunning" ] ; then

    echo "TWS service was already started by another way"
    else

    echo "TWS service was not started"

    echo "Starting service ..."

    ./CardRecognition ../etc/config.ini

    echo "TWS service was exited!"

    fi

    sleep 0.5

    done

