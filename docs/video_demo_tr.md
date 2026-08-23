# 3–5 Dakikalık Türkçe Video Demo Metni

Bu dosya slayt gerektirmeyen, doğrudan ekran kaydı için hazırlanmıştır.

## Kayıttan önce

1. Bilgisayarı yeniden başlatmak veya GPU kullanan diğer uygulamaları kapatmak model bellek hatası riskini azaltır.
2. Foundry Local modellerinin daha önce indirildiğinden ve en az bir test sorusunun çalıştığından emin ol.
3. `demo_data` klasörünün proje kökünde bulunduğunu kontrol et. Klasörde şu iki dosya var:
   - `demo_data/recording_project_info.md`
   - `demo_data/recording_support_policy.md`
4. Kayıt sırasında yalnızca proje kökünde tek bir terminal kullanacağız:

```powershell
# Proje kökünde
.\run.ps1 cli
```

CLI açıldıktan sonra kayıt sırasında şu komutları kullan:

```text
Question> /clean
Question> /upload 'C:\fun_project\microsoft_internship\project\demo_data'
```

`/clean` isteğe bağlıdır; temiz ve tekrarlanabilir bir demo için kullanılır. `/upload` klasördeki desteklenen dosyaları bulur, dosyaları okur, chunk'lara ayırır, embedding'lerini oluşturur ve indeksler.

Not: Model GPU belleği hatası verirse kayıttan önce diğer GPU uygulamalarını kapatıp tekrar dene. Gerekirse `.\run.ps1 cli` komutundan önce `FOUNDRY_REQUIRE_CHAT_GPU=false` ve `FOUNDRY_REQUIRE_EMBEDDING_GPU=false` ayarlarını kullan; CPU modu daha yavaş olabilir.

## Ekran akışı ve konuşma metni

### 0:00–0:25 — Giriş

> Merhaba, ben [adınız]. Bu videoda Microsoft Foundry Local kullanarak geliştirdiğim Local RAG AI Assistant projesini kısaca göstereceğim. Projenin amacı, kullanıcının yerel dokümanlarına dayanarak soru cevaplayabilen, modeller ve dokümanlar bilgisayarda hazır olduktan sonra internet bağlantısı olmadan çalışabilen bir bilgi asistanı geliştirmek.

Önce proje klasörünü ve ardından CLI ekranını göster.

### 0:25–1:05 — Mimari ve çalışma mantığı

Kısa süreliğine `docs/architecture.md` dosyasını veya README içindeki Architecture bölümünü göster. Daha sonra terminalde `.\run.ps1 cli` komutunu çalıştır.

> Sistemin akışı şu şekilde: Kullanıcı CLI üzerinden bir doküman veya doküman klasörü yüklediğinde dosyalar backend'e geliyor. Document loader dosyalardan metni çıkarıyor, metin daha küçük parçalara bölünüyor ve her parça için Foundry Local embedding modeliyle bir vektör oluşturuluyor. Metin, kaynak bilgisi ve JSON formatındaki embedding SQLite'ta saklanıyor. Canlı vektör araması için aynı bilgiler LanceDB'ye de yazılıyor.
>
> Kullanıcı soru sorduğunda soru da yerel olarak embedding'e dönüştürülüyor. Sistem hem anlamsal vektör araması hem de SQLite FTS5 ile anahtar kelime araması yapıyor. Aday parçalar yeniden sıralanıyor ve yalnızca bulunan bağlam, grounded prompt içinde yerel chat modeline gönderiliyor. Son olarak cevap kaynak etiketi, doğrulama bilgisi ve trace ID ile birlikte kullanıcıya dönüyor.

### 1:05–1:35 — CLI ile dosya yükleme

CLI açıldıktan sonra aşağıdaki komutları yaz:

```text
Question> /clean
Question> /upload 'C:\fun_project\microsoft_internship\project\demo_data'
```

> Şimdi iki küçük yerel doküman içeren bir klasörü CLI üzerinden yüklüyorum. Sistem klasördeki desteklenen dosyaları buluyor, metinlerini çıkarıyor, chunk'lara ayırıyor, embedding'lerini oluşturuyor ve arama indeksine ekliyor. Terminalde kaç dosyanın indekslendiğini ve kaç chunk oluşturulduğunu görebiliyorum. Bu adım chat modeline soru sormuyor; yerel indeksleme pipeline'ını çalıştırıyor.

### 1:35–2:05 — Birinci soru: cevaplanabilir bilgi

Şu soruyu sor:

```text
When is the daily project report generated?
```

Beklenen bilgi: `17:30`. Cevapta `recording_project_info.md` kaynağını görmeyi bekliyorum.

> Bu soru yüklediğim dokümanda bulunan bir bilgiyi soruyor. Sistem ilgili chunk'ı getiriyor, model cevabı sadece bu bağlamı kullanarak oluşturuyor ve hangi kaynağın kullanıldığını gösteriyor. Böylece cevabın yalnızca modelin genel bilgisinden gelmediğini görebiliyoruz.

### 2:05–2:35 — İkinci soru: başka bir cevaplanabilir bilgi

```text
When are support requests reviewed?
```

Beklenen bilgi: `Every Monday at 09:00`. Cevapta `recording_support_policy.md` kaynağını görmeyi bekliyorum.

> Burada farklı bir dokümandan gelen bilgiyi soruyorum. Sistem yine en alakalı chunk'ı buluyor ve cevabın hangi dosyadan geldiğini gösteriyor. Böylece birden fazla dosyanın aynı bilgi tabanında kullanılabildiğini görüyoruz.

### 2:35–3:05 — Üçüncü soru: cevaplanamaz bilgi

Şunu sor:

```text
What is the exact office street address?
```

> Bu bilgi dokümanlarda bulunmadığı için asistanın tahmin yürütmesini istemiyorum. Sistem burada “I do not know based on the available documents.” şeklinde cevap veriyor. Bu, RAG sistemindeki önemli güvenlik davranışlarından biri: bağlamda olmayan bilgiyi uydurmamak.

### 3:05–4:00 — Bu projeyi yaparken ne öğrendim?

> Bu projeyi yaparken ilk olarak RAG'in sadece bir chat modeline soru göndermek olmadığını öğrendim. Başarılı bir cevap için dokümanı doğru parçalara bölmek, doğru embedding oluşturmak ve en alakalı parçaları getirmek gerekiyor.
>
> İkinci olarak, kaynak bilgisinin ve citation kontrolünün çok önemli olduğunu gördüm. Model doğru bir cümle kursa bile, bu cümlenin hangi dokümana dayandığını gösteremiyorsak cevabı denetlemek zorlaşıyor.
>
> Ayrıca local AI çalıştırmanın bazı pratik zorluklarını öğrendim. Model boyutu, GPU belleği, CPU ve GPU seçimi, cevap süresi ve ilk model indirme süreci uygulamanın davranışını doğrudan etkiliyor.
>
> Son olarak, answerable, unanswerable ve empty-input gibi farklı test senaryoları hazırlamanın önemini öğrendim. Unit testler ve trace kayıtları sayesinde sadece “çalışıyor” demek yerine retrieval, citation ve hata davranışını da kontrol edebildim.

### 4:00–4:20 — Kapanış

> Özetlemek gerekirse bu proje; yerel doküman yükleme, chunking, embedding, SQLite ve vektör arama, grounded prompt, Foundry Local ile cevap üretme ve citation doğrulama adımlarını tek bir uygulamada birleştiriyor. En önemli kazanımım, bir RAG uygulamasının uçtan uca nasıl tasarlanacağını ve doğrulanacağını öğrenmek oldu. Dinlediğiniz için teşekkür ederim.

## Kayıt sırasında gösterilecek minimumlar

- Proje klasörü ve CLI ekranı
- `docs/architecture.md` veya README Architecture bölümü
- `demo_data` klasörünün `/upload` komutuyla yüklenmesi
- İki cevaplanabilir soru ve kaynak gösterimi
- Bir cevaplanamaz soru ve `I do not know` davranışı
- Kısa “ne öğrendim?” bölümü

## Kayıt güvenliği

- Gerçek kişisel veya gizli doküman yükleme; yalnızca bu klasördeki demo dosyalarını kullan.
- API anahtarı, token, kişisel dosya yolu veya özel müşteri verisi görünürse kaydı durdurup ekranı temizle.
- Model yükleme süresini videoya dahil etmek zorunda değilsin; kayıttan önce modelleri hazırla.
