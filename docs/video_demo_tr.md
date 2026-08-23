# 3–5 Dakikalık Türkçe Video Demo Metni

Bu dosya slayt gerektirmeyen, doğrudan ekran kaydı için hazırlanmıştır.

## Kayıttan önce

1. Bilgisayarı yeniden başlatmak veya GPU kullanan diğer uygulamaları kapatmak model bellek hatası riskini azaltır.
2. Proje klasöründe bir kez `.\run.ps1 status` çalıştırıp veritabanı ve vektör sayılarının göründüğünü kontrol et.
3. Foundry Local modellerinin daha önce indirildiğinden ve en az bir test sorusunun çalıştığından emin ol.
4. İki demo dosyasını hazır tut:
   - `demo_data/recording_project_info.md`
   - `demo_data/recording_support_policy.md`
5. API ve arayüzü iki ayrı terminalde başlat:

```powershell
# Terminal 1 — proje kökünde
.\run.ps1 api

# Terminal 2 — frontend klasöründe
npm run dev
```

Tarayıcıda `http://localhost:5173` adresini aç.

Not: Model GPU belleği hatası verirse kayıttan önce diğer GPU uygulamalarını kapatıp tekrar dene. Gerekirse API terminalini başlatmadan önce `FOUNDRY_REQUIRE_CHAT_GPU=false` ve `FOUNDRY_REQUIRE_EMBEDDING_GPU=false` ayarlarını kullan; CPU modu daha yavaş olabilir.

## Ekran akışı ve konuşma metni

### 0:00–0:25 — Giriş

> Merhaba, ben [adınız]. Bu videoda Microsoft Foundry Local kullanarak geliştirdiğim Local RAG AI Assistant projesini kısaca göstereceğim. Projenin amacı, kullanıcının yerel dokümanlarına dayanarak soru cevaplayabilen, modeller ve dokümanlar bilgisayarda hazır olduktan sonra internet bağlantısı olmadan çalışabilen bir bilgi asistanı geliştirmek.

Ekranda proje klasörünü veya uygulamanın ana ekranını göster.

### 0:25–1:05 — Mimari ve çalışma mantığı

`docs/architecture.md` dosyasını aç veya README içindeki Architecture bölümünü göster.

> Sistemin akışı şu şekilde: Kullanıcı arayüzden bir doküman yüklediğinde dosya önce backend'e geliyor. Document loader dosyadan metni çıkarıyor, metin daha küçük parçalara bölünüyor ve her parça için Foundry Local embedding modeliyle bir vektör oluşturuluyor. Metin, kaynak bilgisi ve JSON formatındaki embedding SQLite'ta saklanıyor. Canlı vektör araması için aynı bilgiler LanceDB'ye de yazılıyor.
>
> Kullanıcı soru sorduğunda soru da yerel olarak embedding'e dönüştürülüyor. Sistem hem anlamsal vektör araması hem de SQLite FTS5 ile anahtar kelime araması yapıyor. Aday parçalar yeniden sıralanıyor ve yalnızca bulunan bağlam, grounded prompt içinde yerel chat modeline gönderiliyor. Son olarak cevap kaynak etiketi, doğrulama bilgisi ve trace ID ile birlikte kullanıcıya dönüyor.

### 1:05–1:35 — Dosya yükleme

Arayüzde `Add a document` alanına tıkla ve iki demo dosyasını seç.

> Şimdi iki küçük yerel doküman yüklüyorum. Burada önemli nokta, yükleme sırasında dosyanın önce okunması, chunk'lara ayrılması, embedding'lerinin oluşturulması ve arama indeksine eklenmesi. Arayüzde kaç dokümanın ve kaç chunk'ın hazır olduğunu görebiliyorum. Bu işlem cloud tabanlı bir chat çağrısı değil; indeksleme yerel pipeline üzerinden yapılıyor.

### 1:35–2:15 — Cevaplanabilir soru

Şu soruyu sor:

```text
When is the daily project report generated?
```

Beklenen cevap: `17:30` ve `recording_project_info.md` kaynağı.

> Bu soru yüklediğim dokümanda bulunan bir bilgiyi soruyor. Sistem ilgili chunk'ı getiriyor, model cevabı sadece bu bağlamı kullanarak oluşturuyor ve hangi kaynağın kullanıldığını gösteriyor. Böylece cevabın yalnızca modelin genel bilgisinden gelmediğini görebiliyoruz.

İkinci soru olarak şunu sorabilirsin:

```text
When are support requests reviewed?
```

Beklenen cevap: `Every Monday at 09:00` ve `recording_support_policy.md` kaynağı.

### 2:15–2:50 — Cevaplanamaz soru ve güvenli davranış

Şunu sor:

```text
What is the exact office street address?
```

> Bu bilgi dokümanlarda bulunmadığı için asistanın tahmin yürütmesini istemiyorum. Sistem burada “I do not know based on the available documents.” şeklinde cevap veriyor. Bu, RAG sistemindeki önemli güvenlik davranışlarından biri: bağlamda olmayan bilgiyi uydurmamak.

### 2:50–3:45 — Bu projeyi yaparken ne öğrendim?

> Bu projeyi yaparken ilk olarak RAG'in sadece bir chat modeline soru göndermek olmadığını öğrendim. Başarılı bir cevap için dokümanı doğru parçalara bölmek, doğru embedding oluşturmak ve en alakalı parçaları getirmek gerekiyor.
>
> İkinci olarak, kaynak bilgisinin ve citation kontrolünün çok önemli olduğunu gördüm. Model doğru bir cümle kursa bile, bu cümlenin hangi dokümana dayandığını gösteremiyorsak cevabı denetlemek zorlaşıyor.
>
> Ayrıca local AI çalıştırmanın bazı pratik zorluklarını öğrendim. Model boyutu, GPU belleği, CPU ve GPU seçimi, cevap süresi ve ilk model indirme süreci uygulamanın davranışını doğrudan etkiliyor.
>
> Son olarak, answerable, unanswerable ve empty-input gibi farklı test senaryoları hazırlamanın önemini öğrendim. Unit testler ve trace kayıtları sayesinde sadece “çalışıyor” demek yerine retrieval, citation ve hata davranışını da kontrol edebildim.

### 3:45–4:10 — Kapanış

> Özetlemek gerekirse bu proje; yerel doküman yükleme, chunking, embedding, SQLite ve vektör arama, grounded prompt, Foundry Local ile cevap üretme ve citation doğrulama adımlarını tek bir uygulamada birleştiriyor. En önemli kazanımım, bir RAG uygulamasının uçtan uca nasıl tasarlanacağını ve doğrulanacağını öğrenmek oldu. Dinlediğiniz için teşekkür ederim.

## Kayıt sırasında gösterilecek minimumlar

- Uygulamanın ana ekranı
- `docs/architecture.md` veya README Architecture bölümü
- İki dosyanın gerçek arayüz üzerinden yüklenmesi
- Bir cevaplanabilir soru ve kaynak gösterimi
- Bir cevaplanamaz soru ve `I do not know` davranışı
- Kısa “ne öğrendim?” bölümü

## Kayıt güvenliği

- Gerçek kişisel veya gizli doküman yükleme; yalnızca bu klasördeki demo dosyalarını kullan.
- API anahtarı, token, kişisel dosya yolu veya özel müşteri verisi görünürse kaydı durdurup ekranı temizle.
- Model yükleme süresini videoya dahil etmek zorunda değilsin; kayıttan önce modelleri hazırla.
