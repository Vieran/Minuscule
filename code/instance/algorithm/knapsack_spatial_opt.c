/* knapsack problem --- spatial optimization version
 * use only one dimension for dp, because max value only relates to dp[i-1][0~k-1]
 */

#include <stdio.h>
#include <stdlib.h>
#define items 4
#define max_weight 8
int dp[max_weight + 1]={0};
int item[items + 1];
int w[items + 1] = {0, 2, 3, 4, 5};
int v[items + 1] = {0, 3, 4, 5, 6};

// find the max value of the backpack
void find_maxv() {
	for (int i = 1; i < items + 1; i++) {
		for (int k = max_weight; k >= w[i]; k--) {
			if (w[i] <= k)
				dp[k] = dp[k] > (dp[k-w[i]]+v[i]) ? dp[k] : (dp[k-w[i]]+v[i]);
		}
	}
}

int main() {
	// initialize weights and values
	
	// find the max value in backpack
	find_maxv();
	printf("max value is %d\n", dp[max_weight]);

	return 0;
}
