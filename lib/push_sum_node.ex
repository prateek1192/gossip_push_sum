defmodule Push_Sum_node do
  use GenServer

  def start_link(initial_state) when is_list(initial_state) do
    GenServer.start_link(__MODULE__, initial_state, debug: [:trace])
  end

  def init(initial_state) do
    {:ok, initial_state}
  end

 def handle_cast({:define_neighbors, neighbors_list},initial_state) do
        {:noreply, [neighbors_list | initial_state]}
 end

def handle_cast({:receive_message, message}, initial_state) do
   
  counter = Enum.at(initial_state,3)
  new_weight = (Enum.at(initial_state,2) + Enum.at(message,2))/2 
  new_sum = (Enum.at(initial_state,1) + Enum.at(message,1))/2
  if Enum.at(initial_state,3) < 3 do
    if check_convergence(initial_state, message) == true do
      counter = counter + 1 
    else
      if Enum.at(initial_state,3) > 0  do 
        counter = 0
      end
    end

    temp_list = [Enum.at(initial_state,0), new_sum, new_weight , counter, false]
    new_list = send_to_random_neighbour(Enum.at(initial_state,0),["ChitChat",new_sum, new_weight])    
    {:noreply,temp_list}
  else 
    Process.exit(self(),:normal)
    {:noreply,initial_state}
    end
  end

defp check_convergence(initial_state, message) do
  prev_ratio = Enum.at(initial_state,1)/Enum.at(initial_state,2)
  new_weight = (Enum.at(initial_state,2) + Enum.at(message,2))/2 
  new_sum = (Enum.at(initial_state,1) + Enum.at(message,1))/2
  new_ratio = new_sum/new_weight 
  counter = Enum.at(initial_state,3)
  abs(prev_ratio-new_ratio) < :math.pow(10,-10)
end

defp send_to_random_neighbour(neighbor_list, message) do     
  if List.first(neighbor_list) == nil do
    neighbor_list
  else
    node_to_send = Enum.at(neighbor_list ,:rand.uniform(length(neighbor_list))-1)
    #node_to_send = Enum.random(neighbor_list)
    if (Process.alive?(node_to_send) == true) do
      GenServer.cast( node_to_send ,{:receive_message, message})
      neighbor_list
    else
    new_list = neighbor_list -- [node_to_send]
    send_to_random_neighbour(new_list,message)
    end
  end
end

end
