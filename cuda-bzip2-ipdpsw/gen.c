#include<stdio.h>
#include<malloc.h>

int main() { 
	int a = 1; 
	FILE *fp = fopen("out.txt","w");
	printf("%d\n",a);
	fprintf(fp,"%d",a);
	return 0;
}
