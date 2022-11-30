import os 
import json
import argparse 
import random 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir',type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--eval_k', type=int, default=10, help='number of instances to display')
    parser.add_argument('--intrusion_k', type=int, default=4, help='the number of instances to serve as background.')
    parser.add_argument('--max_instances', type=int, default=400)
    parser.add_argument('--n_splits', type=int, default=2)
    # parser.add_argument('--min_clus', type=int, default=10)
    args = parser.parse_args() 

    with open(os.path.join(args.checkpoint_dir, 'test_unknown_clusters.json'),'r') as f:
        clusters = json.load(f) 

    cluster_eval = []
    cluster2showninstances = {} 
    
    print(f'{len(clusters)} clusters predicted')
    for cluster_id, instances in clusters.items():
        instances_to_show = instances[:args.eval_k] + instances[-args.eval_k:]
        
        ins_as_text = [] 
        for ins in instances_to_show:
            ins_as_text.append(f"{ins['trigger']}: {ins['sentence']}")
        
        cluster_dict = {
            "id": cluster_id,
            "text": "\n\n".join(ins_as_text),
            "label" : []
        }
        cluster2showninstances[cluster_id] = instances_to_show # List
        cluster_eval.append(cluster_dict)
    
    with open(os.path.join(args.output_dir,'cluster_eval.json'),'w') as f:
        json.dump(cluster_eval, f, indent=2)


    instance_eval = []
    ins2label = {} 
    ins_id = 0
    for cluster_id in cluster2showninstances:
        for ins in cluster2showninstances[cluster_id]:
            text = []
            text.append(f"{ins['trigger']}: {ins['sentence']}")
            text.append("Background Instances:")
            pool = []
            if random.random() < 0.5:
                # sample from same cluster 
                while len(pool) < args.intrusion_k:
                    background_ins = random.sample(clusters[cluster_id], k=1)[0]
                    if background_ins != ins:
                        pool.append(background_ins)
                        text.append(f"{background_ins['trigger']}: {background_ins['sentence']}")
                label = True
            else:
                while len(pool) < args.intrusion_k:
                    sampled_clus = random.sample(list(cluster2showninstances.keys()),k=1)[0]
                    if sampled_clus != cluster_id:
                        background_ins = random.sample(clusters[sampled_clus],k=1)[0]
                        text.append(f"{background_ins['trigger']}: {background_ins['sentence']}")
                        pool.append(background_ins)
                
                label=False 

            ins2label[ins_id] = label 
            instance_eval.append({
                'id': ins_id,
                "text": "\n\n".join(text),
                "label": []
            })
            ins_id +=1 

        if ins_id >= args.max_instances:
            break 

    
    with open(os.path.join(args.output_dir, 'instance_gold.json'),'w') as f:
        json.dump(ins2label, f, indent=2) 
    
    if args.n_splits> 1:
        print(f'splitting into {args.n_splits} for evaluation')
        samples_in_split = args.max_instances // args.n_splits 
        for split_i in range(args.n_splits):
            if split_i < args.n_splits -1:# not last split 
                split_instances = instance_eval[split_i * samples_in_split: (split_i+1)* samples_in_split]
            else:
                split_instances = instance_eval[split_i * samples_in_split: ]
            
            with open(os.path.join(args.output_dir, f'instance_eval_split{split_i}.json'), 'w') as f:
                json.dump(split_instances, f, indent=2) 


    else:
        with open(os.path.join(args.output_dir, 'instance_eval.json'), 'w') as f:
            json.dump(instance_eval, f, indent=2) 

    

    


