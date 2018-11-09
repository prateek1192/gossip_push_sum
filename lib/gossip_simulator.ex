defmodule GossSim do
use Supervisor
  
  def main(args) do
    {:ok, super_pid} = GossSim.start_link(args)
    IO.inspect (super_pid)
    IO.inspect (Supervisor.which_children(super_pid)), label: "The child processes are"
    created_node_pids = for n <- Supervisor.which_children(super_pid), do: elem(n, 1)

    parse_args_and_start(args, created_node_pids)

    final_node_pids = for n <- Supervisor.which_children(super_pid), do: elem(n, 1) 
      IO.inspect (final_node_pids)
  end 

  def start_link(args) do
    Supervisor.start_link(__MODULE__, args)
  end

  def start_child(type) do
    if type == "gossip" do
      Gossip_node
    else
      GenServer.start_link(Gossip_node, [0, true])
    end
  end


  def get_child_list(child_list, 0, algorithm), do: child_list
  
  def get_child_list(child_list, num_of_nodes, algorithm) do
  if algorithm == "gossip" do
    child = 
      %{  
      id: :rand.uniform(num_of_nodes*100000),
      start: {Gossip_node, :start_link, [[0,true]]}
    }
  else
    child = 
      %{  
      id: :rand.uniform(num_of_nodes*100000),
      start: {Push_Sum_node, :start_link, [[0,true]]}
    }

  end

    if num_of_nodes > 0 do
       
      get_child_list( [child] ++ child_list, num_of_nodes-1, algorithm)
    end
  end

  def init(args) do
    num_of_nodes = String.to_integer( Enum.at(args,0))


    
    children = get_child_list([], num_of_nodes, Enum.at(args, 2))
        
    Supervisor.init(children, strategy: :one_for_all)
    

  end
  
  def parse_args_and_start([]) do
    IO.puts "No arguments given."
  end

  def parse_args_and_start(args, created_node_pids) do
    num_of_nodes = String.to_integer( Enum.at(args,0))
    topology = Enum.at(args, 1)
    algorithm = Enum.at(args, 2)
    case topology do
        "full" -> 
          IO.puts("The topology is a full topology.")
          case algorithm do
          "gossip" ->
            IO.puts("The algorithm is gossip Algorithm.")
            #node_pids = Enum.map(1..num_of_nodes, fn x -> create_gossip_node() end)
            node_pids = created_node_pids
            neighbors_list = define_neighbors(node_pids, num_of_nodes, topology)
            IO.inspect node_pids, label: "The before sending node pids are"
            IO.inspect neighbors_list, label: "The before sending neighbors list is"
            send_neighbors(node_pids, neighbors_list)
          "push-sum" ->
           IO.puts("The algorithm is push-sum Algorithm.")
           node_pids = Enum.map(1..num_of_nodes, fn x -> create_push_sum_node([x,1,0,true])  end)
           #node_pids = created_node_pids
           neighbors_list = define_neighbors(node_pids, num_of_nodes, topology)
           IO.inspect node_pids, label: "The before sending node pids are"
           IO.inspect neighbors_list, label: "The before sending neighbors list is"
           send_neighbors(node_pids, neighbors_list)

          end
        "3D" -> 
          IO.puts("The topology is a 3D.")
          case algorithm do
          "gossip" ->
            IO.puts("The algorithm is gossip Algorithm.")
            nearest_cube_root = round(:math.pow(10, 1/3))
            new_num_of_nodes_f = :math.pow(nearest_cube_root, 3)
            new_num_of_nodes = round(new_num_of_nodes_f)
            #node_pids = Enum.map(1..num_of_nodes, fn x -> create_gossip_node() end)
            node_pids = created_node_pids
            neighbors_map = define_neighbors(node_pids, new_num_of_nodes, topology)
            #IO.inspect node_pids, label: "The before sending node pids are"
            #IO.inspect neighbors_list, label: "The before sending neighbors list is"
            send_neighbors(neighbors_map)
          "push-sum" ->
            IO.puts("The algorithm is PUSH SUM Algorithm.")
            nearest_cube_root = round(:math.pow(10, 1/3))
            new_num_of_nodes_f = :math.pow(nearest_cube_root, 3)
            new_num_of_nodes = round(new_num_of_nodes_f)
            node_pids = Enum.map(1..new_num_of_nodes, fn x -> create_push_sum_node([x,1,0,true]) end)
            #node_pids = created_node_pids
            neighbors_map = define_neighbors(node_pids, new_num_of_nodes, topology)
            #IO.inspect node_pids, label: "The before sending node pids are"
            #IO.inspect neighbors_list, label: "The before sending neighbors list is"
            send_neighbors(neighbors_map)
          end
        "rand2D" ->   
          IO.puts("The topology is a rand2D.")
          case algorithm do
          "gossip" ->
            IO.puts("The algorithm is gossip Algorithm.")
            temp = round(Float.ceil(:math.sqrt(num_of_nodes)))
            new_num_of_nodes = round(temp*temp)
            #node_pids = Enum.map(1..num_of_nodes, fn x -> create_gossip_node() end)
            node_pids = created_node_pids
            neighbors_map = define_neighbors(node_pids, new_num_of_nodes, topology)
            #IO.inspect node_pids, label: "The before sending node pids are"
            #IO.inspect neighbors_list, label: "The before sending neighbors list is"
            send_neighbors(neighbors_map)
          "push-sum" ->
            IO.puts("The algorithm is push-sum Algorithm.")
            temp = round(Float.ceil(:math.sqrt(num_of_nodes)))
            new_num_of_nodes = round(temp*temp)
            node_pids = Enum.map(1..new_num_of_nodes, fn x -> create_push_sum_node([x,1,0,true]) end)
            #node_pids = created_node_pids
            neighbors_map = define_neighbors(node_pids, new_num_of_nodes, topology)
            #IO.inspect node_pids, label: "The before sending node pids are"
            #IO.inspect neighbors_list, label: "The before sending neighbors list is"
            send_neighbors(neighbors_map)
           end
        "sphere"  ->   
          IO.puts("The topology is a sphere.")
          case algorithm do
          "gossip" ->
            IO.puts("The algorithm is gossip Algorithm.")
            temp = round(Float.ceil(:math.sqrt(num_of_nodes)))
            new_num_of_nodes = round(temp*temp)
            #node_pids = Enum.map(1..num_of_nodes, fn x -> create_gossip_node() end)
            node_pids = created_node_pids
            neighbors_map = define_neighbors(node_pids, new_num_of_nodes, "torus")
            #IO.inspect node_pids, label: "The before sending node pids are"
            #IO.inspect neighbors_list, label: "The before sending neighbors list is"
            send_neighbors(neighbors_map)
          "push-sum" ->
            IO.puts("The algorithm is push-sum Algorithm.")
            temp = round(Float.ceil(:math.sqrt(num_of_nodes)))
            new_num_of_nodes = round(temp*temp)
            node_pids = Enum.map(1..new_num_of_nodes, fn x -> create_push_sum_node([x,1,0,true]) end)
            #node_pids = created_node_pids
            neighbors_map = define_neighbors(node_pids, new_num_of_nodes, "torus")
            #IO.inspect node_pids, label: "The before sending node pids are"
            #IO.inspect neighbors_list, label: "The before sending neighbors list is"
            send_neighbors(neighbors_map)
          end
        "torus"  ->            
          IO.puts("The topology is a sphere.")
          case algorithm do
          "gossip" ->
            IO.puts("The algorithm is gossip Algorithm.")
            temp = round(Float.ceil(:math.sqrt(num_of_nodes)))
            new_num_of_nodes = round(temp*temp)
            #node_pids = Enum.map(1..num_of_nodes, fn x -> create_gossip_node() end)
            node_pids = created_node_pids
            neighbors_map = define_neighbors(node_pids, new_num_of_nodes, topology)
            #IO.inspect node_pids, label: "The before sending node pids are"
            #IO.inspect neighbors_list, label: "The before sending neighbors list is"
            send_neighbors(neighbors_map)
          "push-sum" ->
            IO.puts("The algorithm is push-sum Algorithm.")
            temp = round(Float.ceil(:math.sqrt(num_of_nodes)))
            new_num_of_nodes = round(temp*temp)
            node_pids = Enum.map(1..new_num_of_nodes, fn x -> create_push_sum_node([x,1,0,true]) end)
            #node_pids = created_node_pids
            neighbors_map = define_neighbors(node_pids, new_num_of_nodes, topology)
            #IO.inspect node_pids, label: "The before sending node pids are"
            #IO.inspect neighbors_list, label: "The before sending neighbors list is"
            send_neighbors(neighbors_map)
          end        
        "line" ->   
          IO.puts("The topology is a line.")
          case algorithm do
          "gossip" ->
            IO.puts("The algorithm is gossip Algorithm.") 
            #node_pids = Enum.map(1..num_of_nodes, fn x -> create_gossip_node() end)
            node_pids = created_node_pids
            neighbors_list = define_neighbors(node_pids, num_of_nodes, topology)
            IO.inspect node_pids, label: "The before sending node pids are"
            IO.inspect neighbors_list, label: "The before sending neighbors list is"
            send_neighbors(node_pids, neighbors_list)
          "push-sum" ->
            node_pids = Enum.map(1..num_of_nodes, fn x -> create_push_sum_node([x,1,0,true]) end)
            #node_pids = created_node_pids
            neighbors_list = define_neighbors(node_pids, num_of_nodes, topology)
            IO.inspect node_pids, label: "The before sending node pids are"
            IO.inspect neighbors_list, label: "The before sending neighbors list is"
            send_neighbors(node_pids, neighbors_list)
          end
        "imp2D" ->   
          IO.puts("The topology is a imp2D.") 
          case algorithm do
          "gossip" ->
            IO.puts("The algorithm is gossip Algorithm.")
            node_pids = created_node_pids
            neighbors_list = define_neighbors(node_pids, num_of_nodes, topology)
            #IO.inspect node_pids, label: "The before sending node pids are"
            #IO.inspect neighbors_list, label: "The before sending neighbors list is"
            send_neighbors(node_pids, neighbors_list)
          "push-sum" ->
            IO.puts("The algorithm is push-sum Algorithm.")
            node_pids = Enum.map(1..num_of_nodes, fn x -> create_push_sum_node([x,1,0,true]) end)
            #node_pids = created_node_pids
            neighbors_list = define_neighbors(node_pids, num_of_nodes, topology)
            #IO.inspect node_pids, label: "The before sending node pids are"
            #IO.inspect neighbors_list, label: "The before sending neighbors list is"
            send_neighbors(node_pids, neighbors_list)
          end
    end  	
  random_node = node_pids |> Enum.shuffle |> hd
  IO.puts("The randomly selected node is ")
  IO.inspect(random_node)
  if algorithm == "gossip" do
    start_gossip(random_node, "ChitChat")
  else
    start_push_sum(random_node, "ChitChat")
  end

  start_time = System.system_time(:millisecond)
  
  mon_list  = Enum.map(node_pids, fn i -> Process.monitor(i) end )
  num_of_nodes_for_monitoring = Enum.count(node_pids)
  if (algorithm == "gossip") do 
    monitor_gossip_processes(0, num_of_nodes_for_monitoring) 
  else
    monitor_pushsum_processes()
  end

  IO.puts "Time for convergence"
  IO.inspect(System.system_time(:millisecond)- start_time )
  end

  def monitor_pushsum_processes() do
    receive do
     {:DOWN, ref, :process, pid, :normal} -> :ok  
     {:DOWN, ref, :process, pid, msg} -> 
       IO.inspect pid, label: "Something went wrong with this process"
    end
  end

  def monitor_gossip_processes(counter,n) do
    receive do
     {:DOWN, ref, :process, pid, msg} -> :ok 
     IO.inspect pid, label: "Process shutdown successfully"
    end
  if (counter+1 < n ) do
    IO.inspect ["Number of dead processes:",counter+1]
    monitor_gossip_processes(counter+1,n) 
    end
  end

  def create_gossip_node() do
    {:ok, pid} = GenServer.start_link(Gossip_node, [0, true])
    pid
  end

  def create_push_sum_node(initial_state) do
    {:ok, pid} = GenServer.start_link(Push_Sum_node, initial_state)
    pid
  end

  def define_neighbors(node_pids, num_of_nodes, topology) do
    IO.puts("Defining the neighbors for topology")
    IO.puts(topology)
    case topology do
      "line" ->
        IO.puts("In define neighbors topology line")
        neighbor_list_reverse = create_line_neighbors(num_of_nodes, node_pids) 
        # IO.inspect neighbor_list_reverse, label: "The final neighbor list reversed is"
        # IO.inspect node_pids, label: "The node pids are"
        neighbor_list = Enum.reverse(neighbor_list_reverse)
        #IO.inspect neighbor_list, label: "The final neighbor list in define neighbor is"
      "full" ->
        IO.puts("In define neighbors topology full")
        neighbor_list_reverse = create_full_neighbors(num_of_nodes, node_pids)
        neighbor_list = Enum.reverse(neighbor_list_reverse)
        #IO.inspect neighbor_list, label: "The final neighbor list is"
      "imp2D" ->
        IO.puts("In define neighbors topology imp2D")
        neighbor_list_reverse = create_imp2D_neighbors(num_of_nodes, node_pids)
        neighbor_list = Enum.reverse(neighbor_list_reverse)
      "rand2D" ->
        IO.puts("In define neighbors topology rand2D")
        map = create_rand2D_neighbors(num_of_nodes, node_pids)
        IO.inspect map, label: "The rand2D map is"
      "torus" ->
        IO.puts("In define neighbors topology torus")
        map = create_torus_neighbors(num_of_nodes, node_pids)
        IO.inspect map, label: "The torus map is"
      "3D" ->
        IO.puts("In define neighbors topology 3D")
        map = create_3D_neighbors(num_of_nodes, node_pids)
        IO.inspect map, label: "The torus map is"
    end
  end
############################################################################################################################################################################
  # Methods to create neighbors for topolgies 
  def create_line_neighbors(num_of_nodes, node_pids) do
    IO.puts("Starting create line neighbors. Number of nodes is")
    IO.puts(num_of_nodes)
    neighbor_list = create_line_neighbors(num_of_nodes, node_pids, [], 0)
    IO.inspect neighbor_list, label: "The neighbor list in top level line create is"
  end
  def create_line_neighbors(num_of_nodes, node_pids, neighbor_list, i) do
    one_less = num_of_nodes-1
    case i do
    0 ->
      if num_of_nodes > 1 do
        create_line_neighbors(num_of_nodes, node_pids, [[Enum.at(node_pids,1)]|neighbor_list], 1)
        # IO.inspect neighbor_list, label: "The neighbor list at 0 is"
      end
    # 1 ->
    #  IO.puts("In nodeid 1")
    #  if num_of_nodes > 1 do
    #    create_line_neighbors(num_of_nodes, node_pids, [[Enum.at(node_pids, 0)]],2)
    #  end
    ^one_less ->
      #IO.puts(i)
      #IO.inspect Enum.at(node_pids, one_less), label: "line create final val"
      [[Enum.at(node_pids,i-1)]|neighbor_list]
      #IO.inspect neighbor_list, label: "The neighbor list in max nodes is"
    _ ->
      IO.puts("In all other case with is i as ")
      IO.puts(i)
      create_line_neighbors(num_of_nodes, node_pids, [[Enum.at(node_pids, i-1), Enum.at(node_pids, i+1)]|neighbor_list],i+1)
      # IO.inspect neighbor_list, label: "The neighbor list is"
    end 
  end
  
  def create_full_neighbors(num_of_nodes, node_pids) do
    IO.puts("Staring create full neighbours. Number of nodes is")
    IO.puts(num_of_nodes)
    create_full_neighbors(num_of_nodes, node_pids, [],0)   
  end

  def create_full_neighbors(num_of_nodes, node_pids, neighbor_list, i) do
    one_less = num_of_nodes-1
    case i do
      ^one_less ->
        IO.puts(i)
        [List.delete_at(node_pids, i-1) | neighbor_list]
      _->
        IO.puts("In all other case with is i as ")
        IO.puts(i)
        create_full_neighbors(num_of_nodes, node_pids, [List.delete_at(node_pids, i) | neighbor_list], i+1)
    end
  end


  def create_torus_neighbors(numNodes, nodes_list) do
   dim = round(:math.floor(:math.sqrt(numNodes)))
   grid2D = Enum.chunk_every(nodes_list, dim)
   map = Enum.reduce(0..dim*dim-1, %{}, fn (x, acc) ->
        if x < numNodes do
        i = round(:math.floor(x/dim)); j = rem(x,dim);
        x_left = if j-1 < 0 do j-1 + length(Enum.at(grid2D, i)) else j-1 end
        x_right = if j+1 >= length(grid2D) do j+1-length(grid2D) else j+1 end
        y_top = if i-1 < 0 do i - 1 + length(grid2D) else i-1 end
        y_bottom = if i+1 >= length(grid2D) do i+1-length(grid2D) else i+1 end
        Map.put(acc, Enum.at(nodes_list, x),
          _2DHelper([y_top, y_bottom, x_left, x_right], i, j, grid2D)
            )
        else 
          acc
        end
      end
      )
    map
  end


  def create_imp2D_neighbors(num_of_nodes, node_pids) do
    IO.puts("Starting create imd2D neighbors. Number of nodes is")
    IO.puts(num_of_nodes)
    neighbor_list = create_imp2D_neighbors(num_of_nodes, node_pids, [], 0)
    # IO.inspect neighbor_list, label: "The neighbor list in top level line create is"
  end
  def create_imp2D_neighbors(num_of_nodes, node_pids, neighbor_list, i) do
    one_less = num_of_nodes-1
    case i do
    0 ->
      if num_of_nodes > 1 do
        nodes_to_exclude = [Enum.at(node_pids, 0), Enum.at(node_pids,1)]
        rand_n = random_neighbour(node_pids, nodes_to_exclude)
        create_imp2D_neighbors(num_of_nodes, node_pids, [[Enum.at(node_pids,1), rand_n]|neighbor_list], 1)
        # IO.inspect neighbor_list, label: "The neighbor list at 0 is"
      end
    ^one_less ->
      nodes_to_exclude = [Enum.at(node_pids, i-1), Enum.at(node_pids,i)]
      rand_n = random_neighbour(node_pids, nodes_to_exclude)
      [[Enum.at(node_pids,i-1), rand_n]|neighbor_list]
    _ ->
      IO.puts("In all other case with is i as ")
      IO.puts(i)
      nodes_to_exclude = [Enum.at(node_pids, i-1), Enum.at(node_pids,i+1), Enum.at(node_pids, i)]
      rand_n = random_neighbour(node_pids, nodes_to_exclude)
      create_imp2D_neighbors(num_of_nodes, node_pids, [[Enum.at(node_pids, i-1), Enum.at(node_pids, i+1), rand_n]|neighbor_list],i+1)
      # IO.inspect neighbor_list, label: "The neighbor list is"
    end
  end


  def create_rand2D_neighbors(num_of_nodes, node_pids) do
    # process_ok_tuples_list = create_server_list(numNodes, protocol)
    # nodes_list = Enum.map(process_ok_tuples_list, fn ({x,y}) -> y end)
    # IO.puts(inspect(nodes_list))
    generate_x_y = Enum.reduce(node_pids, %{},
                            fn (x, acc) -> Map.put(acc, x,
                             {:rand.uniform(10)/10, :rand.uniform(10)/10})
                            end
                              )
    succeding_pid_list =  Enum.map(node_pids, fn x ->
                                  {x , list_generate_helper(node_pids, x)}
                                  end
                                )
    #IO.puts(inspect(succeding_pid_list))
    map = Enum.reduce(succeding_pid_list , %{},
                                  fn ({x, succeding_pid_list_of_x}, acc) ->
                                    IO.puts("pid= #{inspect(x)} and {x,y} = #{inspect(Map.get(generate_x_y, x))}")
                                    new_list = Enum.filter(succeding_pid_list_of_x,
                                              fn (y) ->
                                              #  { _ , point1} = Map.fetch(generate_x_y, x)
                                              #  { _ , point2} = Map.fetch(generate_x_y, y)
                                                check_distance_range(
                                                  Map.get(generate_x_y, x),
                                                  Map.get(generate_x_y, y)
                                                  )
                                              end
                                            )
                                    Map.put(acc, x, new_list)
                                  end
                    )
   map
  end

  #def create_torus_neighbors(numNodes, nodes_list) do
  #  dim = round(:math.floor(:math.pow(numNodes, 1.0/3)))
  #  IO.puts(dim)
  #  grid2D = Enum.chunk_every(nodes_list, dim)
  #  IO.puts(inspect(grid2D))
  #  grid3D = Enum.chunk_every(nodes_list, dim*dim)
  #  IO.puts(inspect(grid3D))
  #  map = Enum.reduce(0..dim*dim-1, %{}, fn (x, acc) ->
  #      if x < numNodes do
  #      i = round(:math.floor(x/dim)); j = rem(x,dim); k = round(:math.floor(x/(dim*dim)))
  #      IO.puts("process = #{x} i = #{i}, j = #{j}, k = #{k}")
  #      x_left = if j-1 < 0 do j-1 + length(Enum.at(grid2D, i)) else j-1 end
  #      x_right = if j+1 >= length(grid2D) do j+1-length(grid2D) else j+1 end
  #      y_top = if i-1 < 0 do i - 1 + length(grid2D) else i-1 end
  #      y_bottom = if i+1 >= length(grid2D) do i+1-length(grid2D) else i+1 end
  #      z_top = if k-1 < 0 do k-1 + length(grid3D) else k-1 end
  #      z_bottom = if k+1 >= length(grid3D) do k+1-length(grid3D) else k+1 end
  #      IO.puts(inspect([y_top, y_bottom, z_top, z_bottom, x_left, x_right]))
  #      Map.put(acc, Enum.at(nodes_list, x),
  #        _2DHelper([y_top, y_bottom, z_top, z_bottom, x_left, x_right], i, j, k, dim, grid2D, grid3D)
  #          )
  #    end
  #  end
  # )
  #  map
  #end

  def _2DHelper([y_top, y_bottom, x_left, x_right], i, j, grid2D) do
    y_top    = Enum.at(Enum.at(grid2D, y_top), j)
    y_bottom = Enum.at(Enum.at(grid2D, y_bottom), j)
    x_left   = Enum.at(Enum.at(grid2D, i), x_left)
    x_right  = Enum.at(Enum.at(grid2D, i), x_right)

    #filtering out nil nodes
    Enum.reduce([y_top, y_bottom, x_left, x_right], [], fn(x, l) ->
        if x == nil do l else [x | l] end
    end
    )
  end
  
  def create_3D_neighbors(numNodes, nodes_list) do
   dim = round(:math.floor(:math.sqrt(numNodes)))
   grid2D = Enum.chunk_every(nodes_list, dim)
   map = Enum.reduce(0..dim*dim-1, %{}, fn (x, acc) ->
        if x < numNodes do
        i = round(:math.floor(x/dim)); j = rem(x,dim);
        x_left = if j-1 < 0 do j-1 + length(Enum.at(grid2D, i)) else j-1 end
        x_right = if j+1 >= length(grid2D) do j+1-length(grid2D) else j+1 end
        y_top = if i-1 < 0 do i - 1 + length(grid2D) else i-1 end
        y_bottom = if i+1 >= length(grid2D) do i+1-length(grid2D) else i+1 end
        Map.put(acc, Enum.at(nodes_list, x),
          _2DHelper([y_top, y_bottom, x_left, x_right], i, j, grid2D)
            )
        else 
          acc
        end
      end
      )
    map
  end
  

############################################################################################################################################################################

  # check if the distance between point is in the range 0.1
  def check_distance_range({x1, y1}, {x2, y2}) do
    #import :math
    #sqrt((x1-x2)(x1-x2) + (y1-y2)(y1-y2)) <= 0.1
    ((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2)) <= 0.01
  end
  
  def list_generate_helper([], element) do
    []
  end

  def list_generate_helper([head | tail], element) do
      if(head == element) do
        list_generate_helper(tail, element)
      else
        [head | list_generate_helper(tail, element)]
      end
  end

  def _2DHelper([y_top, y_bottom, z_top, z_bottom, left, right], i, j, k, dim,  grid2D, grid3D) do
    y_top    = if y_top >= 0 do Enum.at(Enum.at(grid2D, y_top), j) else nil end
    y_bottom = if y_bottom <= 0 do Enum.at(Enum.at(grid2D, y_bottom), j) end
    z_top = Enum.at(Enum.at(grid3D, z_top), j*dim + i)
    z_bottom = Enum.at(Enum.at(grid3D, z_bottom), j*dim + i)
    x_left   = Enum.at(Enum.at(grid2D, i), left)
    x_right  = Enum.at(Enum.at(grid2D, i), right)

    #filtering out nil nodes
    Enum.reduce([y_top, y_bottom, z_top, z_bottom, x_left, x_right], [], fn(x, l) ->
        if x == nil do l else [x | l] end
    end)
  end


  def _2DHelper1([y_top, y_bottom, z_top, z_bottom, x_left, x_right], i, j, k, grid) do
    y_top = if y_top >= 0 do  Enum.at(Enum.at(Enum.at(grid, k), y_top), j) else nil end
    IO.puts("y_top #{inspect y_top}")
    y_bottom = if y_bottom <= length(Enum.at(grid, k)) - 1 do Enum.at(Enum.at(Enum.at(grid, k), y_bottom), j) else nil end
    IO.puts("y_bottom #{inspect(y_bottom)}")
    z_top = if z_top >= 0 do Enum.at(Enum.at(Enum.at(grid, z_top), i),j) else nil end
    IO.puts("z_top #{inspect z_top}")
    z_bottom = if z_bottom <= length(grid) - 1 do Enum.at(Enum.at(Enum.at(grid, z_bottom), i),j) else nil end
    IO.puts("z_bottom #{inspect z_bottom}")
    x_left   = if x_left >= 0 do Enum.at(Enum.at(Enum.at(grid, k), i), x_left) else nil end
    IO.puts("x_left #{inspect x_left}")
    x_right  = if x_right <= length(Enum.at(Enum.at(grid, k), i)) - 1 do Enum.at(Enum.at(Enum.at(grid, k), i), x_right)  else nil end
    IO.puts("x_right #{inspect x_right}")
    #filtering out nil nodes
    Enum.reduce([y_top, y_bottom, z_top, z_bottom, x_left, x_right], [], fn(x, l) ->
      if x == nil do l else [x | l] end
    #IO.puts(inspect(final_list))
    end)
  end 

  def random_neighbour(node_pids, nodes_to_exclude) do
    list_to_choose_from = node_pids -- nodes_to_exclude
    rand_node = Enum.take_random(list_to_choose_from, 1)
    Enum.at(rand_node, 0)
  end
  def send_neighbors(node_pids, neighbors_list) do
    one_less = Enum.count(node_pids)-1
    for node_number <- 0..one_less do
      GenServer.cast(Enum.at(node_pids, node_number), {:define_neighbors, Enum.at(neighbors_list, node_number)})
    end
  end
  def send_neighbors(neighbors_map) do
    Enum.each neighbors_map,  fn {k, v} ->
        GenServer.cast(k, {:define_neighbors, v})
        IO.inspect(k)
        IO.inspect(v)
    end 
  end
  
  def start_gossip(random_node_pid, message) do
    GenServer.cast(random_node_pid, {:receive_message, message})
  end

  def start_push_sum(random_node_pid, message) do
    message_to_send = [message, 0 , 0, false]
    GenServer.cast(random_node_pid, {:receive_message, message_to_send})
  end
end

