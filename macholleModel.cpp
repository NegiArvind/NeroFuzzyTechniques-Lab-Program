#include<bits/stdc++.h>
using namespace std;
bool compareActualAnswerWithPredicted(int actual[],int predicted[],int totalComb)
{
	for(int i=0;i<totalComb;i++){
		if(actual[i]!=predicted[i])
			return false;
	}
	return true;

}
int main()
{
	int n,i,j,k;
	const int m=32;
	cout<<"Enter number of inputs\n";
	cin>>n;
	int totalComb=pow(2,n);
	int combinationInput[totalComb][n];
	int t=n+1;
	int weightCombination[totalComb][t];
	int actualOutput[totalComb];
    for(i=0;i<totalComb;i++){
    	string binary = bitset<m>(i).to_string();
    	cout<<binary<<"\n";
    	k=0;
    	int positiveWeight=0;
    	for(j=m-n;j<m;j++){
    		combinationInput[i][k]=binary[j]-48;
    		if(binary[j]-48==0){
    			weightCombination[i][k]=1;
    			positiveWeight++;
    		}else{
    			weightCombination[i][k]=-1;
    		}
    		k++;
    	}
    	weightCombination[i][k]=positiveWeight;
    }
    for(i=0;i<totalComb;i++)
    {
    	for(j=0;j<n;j++)
    		cout<<combinationInput[i][j]<<" ";
    	cout<<"\n";
    }

    for(i=0;i<totalComb;i++)
    {
    	for(j=0;j<=n;j++)
    		cout<<weightCombination[i][j]<<" ";
    	cout<<"\n";
    }
    for(i=0;i<totalComb;i++){
    	int result=combinationInput[i][0];
    	for(j=1;j<n;j++){
    		result=result&combinationInput[i][j];
    	}
    	actualOutput[i]=result;
    }
    cout<<"Actual answer\n";
    for(i=0;i<totalComb;i++)
    {
    	cout<<actualOutput[i]<<"\n";
    }
    cout<<"\n";

    int predictedOutput[totalComb];
    for(i=0;i<totalComb;i++)
    {
    	// cout<<n<<" ";
    	int theta=n*weightCombination[i][n]-(n-weightCombination[i][n]);
    	cout<<"theta="<<theta<<"\n";
    	for(j=0;j<totalComb;j++){
    		int sum=0;
    		for(k=0;k<n;k++){
    			sum+=combinationInput[j][k]*weightCombination[i][k];
    		}
    		cout<<"sum="<<sum;
    		if(sum>theta){
    			predictedOutput[j]=1;
    		}else{
    			predictedOutput[j]=0;
    		}
    	}
    	cout<<"predictedOutput\n";
    	for(int r=0;r<totalComb;r++)
    	{
    		cout<<predictedOutput[r]<<"\n";
    	}
    	bool answer=compareActualAnswerWithPredicted(actualOutput,predictedOutput,totalComb);
		if(answer){
			cout<<"\nFor below weight it is giving correct output as expected\n";
			for(int p=0;p<n;p++){
				cout<<weightCombination[i][p]<<" ";
			}
		}
    }
    
}