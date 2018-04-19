package neural

import "log"

func cost(outputs, desired []float64) float64 {
	if len(outputs) != len(desired) {
		log.Fatal("Desired Length doesn't match output length")
	}
	total := 0.
	for i := range outputs {
		total += (outputs[i] - desired[i]) * (outputs[i] - desired[i])
	}
	return total
}

func TotalCost(net Network, inputs, desired [][]float64) float64 {
	c := 0.
	for i, input := range inputs {
		net.SetInput(input)
		net.EvaluateNetwork()
		output := net.GetOutputValues()
		//		fmt.Println(getValues(net[1]))
		//		fmt.Println(getValues(net[2]))
		//		fmt.Println("----")
		c += cost(output, desired[i])
	}
	return c / float64(2*len(inputs))
}
