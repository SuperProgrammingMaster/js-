let datasets = []

for(let i=0; i<1000;i++)
{
    datasets.push({x:i,y:i**2 + 3})
}

function sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
}

function sigmoidDerivative(x) {
    let sig = sigmoid(x)
    return sig * (1 - sig)
}


function ReLU(x) {
    return Math.max(0,x)

}

function LAF(x) {
    return x
}
function Loss(x) {
    return (datasets[x].y-getModelResult(x).전계층의총합)
}

function reluDerivative(x) {
    if(x>0)
    {
        return 1
    }
    return 0
}

let nodes = []


function 노드만들기(size) {
    for(let i=0; i<size.length; i++)
    {
        nodes[i] = []
        for(let k=0; k<size[i]; k++)
        {
            let temp = []
            
            for(let m=0; m<size[i-1]; m++)
            {
                temp.push(Math.random())
                
            }
            nodes[i][k] = ({w:temp,b:Math.random()})
        }
    }
}

노드만들기([1,3,3,1])



function getModelResult(x) {
    let layerOutputs = [] //x값 저장용
    

    for(let l =0; l<nodes.length; l++)
    {
        let temp = []
        for(let k =0; k<nodes[l].length; k++)
        {
            if(l==0)
            {
                temp.push(x)
            }
            else
            {
                let sum = 0
                for(let j=0; j<nodes[l][k].w.length; j++)
                {
                    sum = sum + ReLU(layerOutputs[l-1][j] * nodes[l][k].w[j] + nodes[l][k].b)
                }
                temp.push(sum)
            }
        }
        layerOutputs.push(temp)
    }
    return {output:layerOutputs[nodes.length-1][0],layerOutputs:layerOutputs}
}

function backward(batchsize,maxepoch,learningRate,earlyexit=false) {
    if(batchsize> datasets.length)
    {
        batchsize = datasets.length
    }

    for(let batch =0; batch<batchsize; batch++)
    {
        for(let epoch=0; epoch<maxepoch; epoch++)
        {
            let x = datasets[batch].x;
            let y = datasets[batch].y;

            let {layerOutputs,output} = getModelResult(x)

            console.log(`Loss : ${y - output}`)
            if(earlyexit)
            {
                if(Math.abs(y-output) >5)
                {
                    return
                }
            }
            for(let l =nodes.length-1; l>=0; l--)
            {
                for(let k =0; k<nodes[l].length; k++)
                {
                    let 델타 = 0
                    if(l==nodes.length-1)
                    {
                        델타 = (ReLU(layerOutputs[l][k]) - y) * reluDerivative(layerOutputs[l][k])
                        
                    }
                    else
                    {
                        let m= nodes[l+1].length
                        for(let i=0; i<m; i++)
                        {
                            델타 = 델타 + nodes[l+1][i].d * nodes[l+1][i].w[k]* reluDerivative(layerOutputs[l][k])
                        }
                    }
                    nodes[l][k] = {w:nodes[l][k].w,b:nodes[l][k].b,d:델타}
                    for(let j=0; j<nodes[l][k].w.length; j++)
                    {
                        nodes[l][k].w[j] -=learningRate * nodes[l][k].d * ReLU(layerOutputs[l-1][j]) 
                    }
                    nodes[l][k].b -= learningRate * nodes[l][k].d
                }
                
            }
        }
    }
}


backward(1000,1000,0.00001,true)

const readline = require('readline');

// readline 인터페이스 설정
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

// 사용자에게 입력 요청
console.log("x의 값을 입력하시오")
start()
function start() {
    rl.question('', (input) => {
        if (input === "stop") {
            rl.close();
            return; 
        } else {
            console.log(`정답 : ${getModelResult(input).output}`); 
            start();
        }
    });
}