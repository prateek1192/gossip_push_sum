defmodule Gossip_node do
  use GenServer

def start_link(zero_counter_gossiping_neighbors) when is_list(zero_counter_gossiping_neighbors) do
    GenServer.start_link(__MODULE__, zero_counter_gossiping_neighbors, debug: [:trace])
  end

  def init(intial_state) do
    {:ok, intial_state}
    end

  def handle_cast({:define_neighbors, neighbor_list}, intial_state) do
    {:noreply,[neighbor_list | intial_state]}
  end

  def handle_cast({:receive_message,message}, intial_state) do
    counter = Enum.at(intial_state,1)
    new_counter = counter + 1
    if counter == 5 do
      Process.exit(self(), :kill)
    end
    if counter <= 9 do
      GenServer.cast(self(),{:send_periodic_message})
    else
      Process.exit(self(),:normal);
    end
    new_state = [Enum.at(intial_state,0),new_counter,false] 
    {:noreply,new_state}
    end 

    def handle_cast({:send_periodic_message},intial_state) do

      new_list = send_to_random_neighbour(Enum.at(intial_state,0),["message"])

      if  List.first(new_list) != nil do
        Process.sleep(10)
        GenServer.cast(self(),{:send_periodic_message})
      end
        {:noreply,intial_state}
    end

    def send_message_helper(neighbor_list, message) do

      node_to_send = Enum.at(neighbor_list ,(:rand.uniform(length(neighbor_list))-1))
      if (Process.alive?(node_to_send) == true) do
       GenServer.cast(node_to_send, {:receive_message,message})
       neighbor_list
      else
        new_list = neighbor_list -- [node_to_send]
        send_to_random_neighbour(new_list,message)
      end
    end


    def send_to_random_neighbour(neighbor_list,message) do  
        if List.first(neighbor_list) == nil do
            neighbor_list
        else
            send_message_helper(neighbor_list, message)
        end
    end
end
