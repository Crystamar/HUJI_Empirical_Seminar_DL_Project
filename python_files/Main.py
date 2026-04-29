import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from PatchEmbedding import *
from SepPatchEmbedding import *
from VisionTransformer import VisionTransformer


if __name__ == '__main__':
    cifar10_train = torchvision.datasets.CIFAR10(root="./datasets/cifar10", train=True, transform=transforms.Compose([transforms.Resize(48), transforms.ToTensor()]), download=True)
    cifar10_test = torchvision.datasets.CIFAR10(root="./datasets/cifar10", train=False, transform=transforms.Compose([transforms.Resize(48), transforms.ToTensor()]), download=True)

    # Hyper-parameters
    # betas = (0.9, 0.999)
    lr = 0.001
    do_p = 0
    wd = 0
    batch_size = 100
    d = 256
    encoder_layers = 3
    heads = 8
    ff_dim = 512
    cifar_classes = 10

    cifar10_trainloader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)
    cifar10_testloader = DataLoader(cifar10_test, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    cs = nn.CosineSimilarity(2)

    train_steps = len(cifar10_trainloader)
    test_steps = len(cifar10_testloader)
    epochs = 25

    """
    Non-seperate patch sizes options are:
    
    Unbiased:
    A tuple of the same int x which 48 is divisible by, y times when y = 48/x
    
    Biased:
    3x3: (2, 44, 2), (12, 24, 12), (14, 20, 14), (18, 12, 18), (19, 10, 19), (22, 4, 22)
    4x4: (2, 22, 22, 2), (6, 18, 18, 6), (8, 16, 16, 8), (10, 14, 14, 10), (16, 8, 8, 16), (19, 5, 5, 19), (22, 2, 2, 22)
    6x6: (4, 8, 12, 12, 8, 4), (12, 8, 4, 4, 8, 12)
    
    (I chose not to use a generic class that allows any legal sequence of patch sizes for the non-seperate embeddings,
    because it would necessitate looping in each forward operation which i assumed would slow performance noticeably)
    
    (The seperate embeddings models allow for any legal patch sizes sequence)
    """
    patches = [(2, 44, 2), (12, 24, 12), (14, 20, 14), (16, 16, 16), (18, 12, 18), (19, 10, 19), (22, 4, 22),
               (2, 22, 22, 2), (6, 18, 18, 6), (8, 16, 16, 8), (10, 14, 14, 10), (12, 12, 12, 12), (16, 8, 8, 16), (19, 5, 5, 19), (22, 2, 2, 22),
               (4, 8, 12, 12, 8, 4), (8, 8, 8, 8), (12, 8, 4, 4, 8, 12)]
    accs_3x3 = {p: [] for p in patches[:7]}
    accs_4x4 = {p: [] for p in patches[7:15]}
    accs_6x6 = {p: [] for p in patches[15:]}
    acc_sizes = [accs_3x3, accs_4x4, accs_6x6]
    acc_size_strs = ['3x3', '4x4', '6x6']
    cur_accs = accs_3x3
    use_sep_embeddings = False

    for num, patch in enumerate(patches):
        if num == 7:
            cur_accs = accs_4x4
        if num == 15:
            cur_accs = accs_6x6

        print(f"Patch size/s is: {patch}")
        if use_sep_embeddings:
            pe = SepPatchEmbeddings(patch_sizes=patch, d=d)
        else:
            pe = PatchEmbeddings(patch_sizes=patch, d=d)
        vt = VisionTransformer(embedding=pe, d=d, num_layers=encoder_layers, num_heads=heads, ff_dim=ff_dim, num_classes=cifar_classes, dropout_p=do_p)
        optim = torch.optim.SGD(vt.parameters(), lr=0.01, weight_decay=wd)
        accs = []

        for epoch in range(epochs):
            vt.train()
            # print(f"epoch num {epoch + 1}")
            for i, (inputs, labels) in enumerate(cifar10_trainloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = vt(inputs)
                loss = criterion(outputs, labels)

                optim.zero_grad()
                loss.backward()
                optim.step()

                # if (i + 1) % 100 == 0:
                #     print(f"Step {i + 1}/{train_steps}. Loss is {loss.item()}")

            vt.eval()
            with torch.no_grad():
                n_correct = 0
                n_samples = 0
                for i, (inputs, labels) in enumerate(cifar10_testloader):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = vt(inputs)
                    _, preds = torch.max(outputs, 1)

                    n_samples += batch_size
                    n_correct += (preds == labels).sum()

                    # if (i + 1) % 100 == 0:
                    #     print(f"Step {i + 1}/{test_steps}. Loss is {criterion(outputs, labels)}. Accuracy is {n_correct/n_samples}")

                acc = n_correct/n_samples
                cur_accs[patch].append(acc)
                print(f"Patch sizes: {patch}, Epoch: {epoch+1}, Accuracy: {acc}")

        leng = len(patch)
        with torch.no_grad():
            for i, pos_emb in enumerate(vt.position_embedding[0, 1:]):
                plt.subplot(leng, leng, i+1)
                plt.imshow(torch.reshape(cs(pos_emb, vt.position_embedding[0, 1:].unsqueeze(0)), (leng, leng)))
                plt.axis('off')
            plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
            cax = plt.axes([0.85, 0.1, 0.075, 0.8])
            plt.colorbar(cax=cax)
            patch_str = [str(p) for p in patch]
            plt.suptitle(f"Position embedding cosine similarity ({', '.join(patch_str)})")
            plt.savefig(f"pos_similarity_patch {'_'.join(patch_str)}")
            plt.close()

    for accs_dict, dims in zip(acc_sizes, acc_size_strs):
        for patch, accs in accs_dict.items():
            patch_str = [str(p) for p in patch]
            plt.plot(accs, label=f"({', '.join(patch_str)})")
        plt.xlabel("Epoch num")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy per epoch for {dims} patch sizes with seperate embeddings")
        plt.legend(prop={'size': 6})
        plt.savefig(f"accs_{dims}")
        plt.close()
