package neural

import "testing"

func sumInputs(inputs []float64) float64 {
	total := 0.0
	for _, x := range inputs {
		total += x
	}
	return total
}

func average(inputs []float64) float64 {
	if len(inputs) == 0 {
		return 0
	}
	total := 0.0
	for _, x := range inputs {
		total += x
	}
	return total / float64(len(inputs))
}

func TestEvaluateNeuron(t *testing.T) {
	//setup
	input1 := []float64{0.125, 0.25, 0.5}
	testNeuron1 := neuron{
		f:     sumInputs,
		value: 0,
	}
	testNeuron2 := neuron{f: average}
	//tests
	testNeuron1.EvaluateNeuron(input1)
	if testNeuron1.value != 0.875 {
		t.Errorf("testNeuron value %v, expected %v", testNeuron1.value, 0.875)
	}
	testNeuron1.EvaluateNeuron([]float64{0.0})
	if testNeuron1.value != 0.0 {
		t.Errorf("testNeuron value %v, expected %v", testNeuron1.value, 0.)
	}
	testNeuron2.EvaluateNeuron([]float64{2.0, 4.0})
	if testNeuron2.value != 3.0 {
		t.Errorf("testNeuron2 value %v, expected %v", testNeuron2.value, 2)
	}
}

func TestMakeWeightedSum(t *testing.T) {
	neuron := MakeWeightedSumNeuron([]float64{0.5, 1, 2})
	shortInput := []float64{2.0, 4.0}
	midInput := []float64{2, 4, 6}
	longInput := []float64{2, 4, 6, 8}
	shortExpect := 5.0
	midExpect := 17.0
	longExpect := 17.0

	neuron.EvaluateNeuron(shortInput)
	if neuron.value != shortExpect {
		t.Errorf("short input test, expected %v, actual %v", shortExpect, neuron.value)
	}
	neuron.EvaluateNeuron(midInput)
	if neuron.value != midExpect {
		t.Errorf("min input test, expected %v, actual %v", midExpect, neuron.value)
	}
	neuron.EvaluateNeuron(longInput)
	if neuron.value != longExpect {
		t.Errorf("long input test, expected %v, actual %v", longExpect, neuron.value)
	}
}
