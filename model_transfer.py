import torch
from AttentionNet_for_torchscript import AttentionNet
from parameter import *



def main():
    device = 'cuda'
    model = AttentionNet(INPUT_DIM, EMBEDDING_DIM).to(device)
   
    if False:
        if device == 'cpu':
            checkpoint = torch.load(model_path + '/checkpoint.pth', map_location='cpu')
            print('Loading Model on CPU...')
        else:
            checkpoint = torch.load(model_path + '/checkpoint.pth')
            print('Loading Model on GPU...')

        model.load_state_dict(checkpoint['model'])
        curr_episode = checkpoint['episode']
        print("curr_episode set to ", curr_episode)

    model.eval()

    traced_script_module = torch.jit.script(model)
    traced_script_module.save("model.pt")
    print("Transfer to pt file successfully")
    #model = torch.jit.load('model.pt')
    #output = model(in1 ,in2, in3)
    #print(output.device)

if __name__ == '__main__':
    main()
