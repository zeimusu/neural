package neural

import "testing"

func TestSigmoidNetword(t *testing.T) {
	var networkWeights = [][][]float64{
		{
			{0.2, 0.3},
			{0, 0.5},
			{2.1, 0},
		},
		{
			{1.0, 2.0, 3.0},
			{0, 1, 0},
		},
		{
			{1, 1},
			{0.4, -0.5},
		},
	}

	var networkBias = [][]float64{
		{0.5, 0.2, 0.1},
		{2.2, -0.2},
		{0.5, 0.5},
	}
	var inputs = [][]float64{
		{0.2, 0.6},
		{0.8, 0.1},
		{0.4, 0.8},
		{0.2, -0.1},
	}
	var desired = [][]float64{
		{0, 1},
		{1, 0},
		{0, 1},
		{1, 0},
	}
	testNet := MakeSigmoidNetwork(2, networkWeights, networkBias)
	actual := totalCost(testnet, inputs, desired)
	t.Logf("Actual cost %v\n", actual)
}
