/* knapsack problem --- base version
 * given items[i], each of which weights w[n] and values v[i](i ranges from 0~n-1)
 * and a backpack can store max_weight.
 * find the max value that backpack can fit
 */

#include <stdio.h>
#include <stdlib.h>
#define items 4
#define max_weight 8
int dp[items + 1][max_weight + 1]={0};
int item[items + 1];
int w[items + 1] = {0, 2, 3, 4, 5};
int v[items + 1] = {0, 3, 4, 5, 6};

// find the max value of the backpack
void find_maxv() {
	for (int i = 1; i < items + 1; i++) {
		for (int k = 1; k < max_weight + 1; k++) {
			if (w[i] > k)
				dp[i][k] = dp[i-1][k];
			else
				dp[i][k] = dp[i-1][k] > (dp[i-1][k-w[i]]+v[i]) ? dp[i-1][k] : (dp[i-1][k-w[i]]+v[i]);
		}
	}
}


// reverse and find what the backpack stores
void find_items(int i, int k) {
	if (i == 0) { // print what is in the backpack
		for (int j = 0; j < items + 1; j++)
			if (item[j] == 1) // item[i] in backpack
				printf("%d ", j);
		return;
	}
	if (k < w[i] || dp[i][k] == dp[i-1][k-w[i]]) {
		item[i] = 0;
		find_items(i-1, k);
	} else if (dp[i][k] == dp[i-1][k-w[i]]+v[i]){
		item[i] = 1;
		find_items(i-1, k-w[i]);
	}
}

int main() {
	// initialize weights and values
	
	// find the max value in backpack
	find_maxv();

	// find what is in the backpack
	find_items(items, max_weight);
	return 0;
}
